from flask import Flask, render_template, request, flash, redirect, url_for, session, jsonify
import os
from datetime import datetime
from dotenv import load_dotenv
import paramiko
from io import BytesIO
import requests
import openai
import logging
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import hashlib
import tempfile
import json
from sqlalchemy import create_engine, text
import shutil
import stat
from werkzeug.utils import secure_filename

# 配置日誌
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 載入環境變數
try:
    load_dotenv()
    logger.info("環境變數載入成功")
    logger.debug(f"當前工作目錄: {os.getcwd()}")
    logger.debug(f"OPENAI_API_KEY 是否存在: {'是' if os.getenv('OPENAI_API_KEY') else '否'}")
    logger.debug(f"AI_PROVIDER 設置為: {os.getenv('AI_PROVIDER', 'not set')}")
except Exception as e:
    logger.error(f"載入環境變數時發生錯誤: {str(e)}")

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# 從環境變數獲取遠端配置
REMOTE_HOST = os.getenv('REMOTE_HOST')
REMOTE_USER = os.getenv('REMOTE_USER')
REMOTE_PASSWORD = os.getenv('REMOTE_PASSWORD')
REMOTE_PATH = os.getenv('REMOTE_PATH')
AI_MODEL = os.getenv('AI_MODEL', 'ollama')  # 改為預設使用 ollama
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://10.96.196.63:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1')

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'md'}  # 增加 doc, docx, md 格式

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_date_path():
    """獲取時間戳，不再創建年月日目錄"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return 'uploads', timestamp  # 直接返回 uploads 目錄和時間戳

def get_sftp_connection():
    """建立 SFTP 連接"""
    try:
        transport = paramiko.Transport((REMOTE_HOST, 22))
        transport.connect(username=REMOTE_USER, password=REMOTE_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        return sftp, transport
    except Exception as e:
        print(f"SFTP 連接錯誤：{str(e)}")
        return None, None

def ensure_remote_path(sftp, path):
    """確保遠端路徑存在"""
    try:
        current_path = ''
        for folder in path.split('/'):
            if folder:
                current_path += '/' + folder
                try:
                    sftp.stat(current_path)
                except FileNotFoundError:
                    logger.info(f"創建目錄: {current_path}")
                    sftp.mkdir(current_path)
        return True
    except Exception as e:
        logger.error(f"創建遠端路徑錯誤：{str(e)}")
        return False

def save_file_to_remote(file, remote_path, filename):
    """保存文件到遠端伺服器"""
    try:
        sftp, transport = get_sftp_connection()
        if not sftp:
            return False, "無法連接到遠端伺服器"
        
        try:
            # 確保遠端路徑存在
            try:
                sftp.stat(remote_path)
            except FileNotFoundError:
                sftp.mkdir(remote_path)
                logger.info(f"已創建目錄: {remote_path}")
            
            # 直接使用原始文件名，不添加時間戳
            remote_file_path = os.path.join(remote_path, filename)
            
            # 如果文件已存在，先刪除
            try:
                sftp.remove(remote_file_path)
                logger.info(f"已刪除舊文件: {remote_file_path}")
            except FileNotFoundError:
                pass
            
            # 保存文件
            sftp.putfo(file, remote_file_path)
            logger.info(f"文件已保存到: {remote_file_path}")
            
            return True, "文件上傳成功"
            
        except Exception as e:
            logger.error(f"保存文件時發生錯誤: {str(e)}")
            return False, str(e)
        finally:
            sftp.close()
            transport.close()
            
    except Exception as e:
        logger.error(f"連接伺服器時發生錯誤: {str(e)}")
        return False, str(e)

# 添加數據庫連接設定
CONNECTION_STRING = "postgresql+psycopg2://postgres:test@localhost:5432/vector_db"
COLLECTION_NAME = "knowledge_base"

# 初始化 embeddings 和向量數據庫
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

# 自定義提示模板
RAG_PROMPT = PromptTemplate(
    template="""使用以下上下文來回答問題。如果上下文中沒有相關資訊，或無法找到答案，請明確說明「抱歉，在提供的文件中找不到相關資訊」，不要試圖編造答案。

上下文：{context}

問題：{question}
請用繁體中文回答，並確保：
1. 如果上下文中有相關資訊，請基於上下文回答
2. 如果上下文中沒有相關資訊，請明確說明找不到相關資訊
3. 不要編造任何未在上下文中提到的資訊""",
    input_variables=["context", "question"]
)

def get_ai_response(message, model_type=AI_MODEL, selected_files=None, use_knowledge_base=False):
    """獲取 AI 回應，整合 RAG 功能"""
    try:
        logger.info(f"開始處理請求 - 消息: {message}")
        logger.info(f"模型類型: {model_type}")
        logger.info(f"選中的文件: {selected_files}")
        
        # 如果有選中文件，使用 RAG
        if selected_files:
            all_answers = []
            
            try:
                # 根據模型類型選擇 LLM
                if model_type == 'openai':
                    llm = OpenAI(temperature=0)
                    logger.info("已創建 OpenAI LLM")
                else:  # ollama
                    from langchain_community.llms import Ollama
                    llm = Ollama(
                        base_url=OLLAMA_URL,
                        model=OLLAMA_MODEL,
                        temperature=0
                    )
                    logger.info("已創建 Ollama LLM")
                
                # 創建 embeddings 實例
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=os.getenv('OPENAI_API_KEY')
                )
                logger.info("已創建 embeddings 實例")
                
                # 處理每個選中的文件
                for file_name in selected_files:
                    try:
                        # 使用固定的 UUID 作為 collection_id
                        collection_id = "f0d144a4-3d46-496e-b5b2-2d7684b21274"
                        logger.info(f"處理文件 {file_name} (collection: {collection_id})")
                        
                        try:
                            # 創建向量存儲
                            vector_store = PGVector(
                                collection_name=collection_id,
                                connection_string=CONNECTION_STRING,
                                embedding_function=embeddings
                            )
                            logger.info(f"已創建向量存儲: {collection_id}")
                            
                            # 直接從數據庫獲取文檔內容
                            engine = create_engine(CONNECTION_STRING)
                            with engine.connect() as connection:
                                # 先檢查表中的所有數據
                                check_query = text("""
                                    SELECT collection_id, document 
                                    FROM langchain_pg_embedding;
                                """)
                                all_docs = connection.execute(check_query).fetchall()
                                logger.info(f"數據庫中的所有文檔: {all_docs}")
                                
                                if all_docs:
                                    # 使用第一個找到的文檔
                                    doc_id, doc_content = all_docs[0]
                                    logger.info(f"使用文檔 ID: {doc_id}, 內容: {doc_content}")
                                    
                                    # 創建提示
                                    prompt = f"""基於以下文檔內容回答問題：

文檔內容：{doc_content}

問題：{message}

請直接使用文檔中的資訊回答，不要添加任何額外資訊。
請用繁體中文回答。"""

                                    # 使用 Ollama 生成回答
                                    response = requests.post(
                                        f"{OLLAMA_URL}/api/generate",
                                        json={
                                            "model": OLLAMA_MODEL,
                                            "prompt": prompt,
                                            "stream": False
                                        }
                                    )
                                    
                                    if response.status_code == 200:
                                        answer = response.json()['response']
                                    else:
                                        answer = f"API 請求失敗: {response.text}"
                                    
                                    formatted_answer = f"基於文件 '{file_name}' 的回答：\n{answer}"
                                    all_answers.append(formatted_answer)
                                    logger.info(f"已生成文件 {file_name} 的回答")
                                else:
                                    logger.error("數據庫中沒有找到任何文檔")
                                    all_answers.append(f"處理文件 '{file_name}' 時發生錯誤: 找不到文檔內容")
                            
                        except Exception as e:
                            logger.error(f"處理向量存儲時發生錯誤: {str(e)}")
                            continue
                            
                    except Exception as e:
                        logger.error(f"處理文件 {file_name} 時發生錯誤: {str(e)}")
                        all_answers.append(f"處理文件 '{file_name}' 時發生錯誤: {str(e)}")
                
                # 合併所有答案
                if all_answers:
                    final_answer = "\n\n".join(all_answers)
                    if model_type != 'openai':
                        final_answer += "\n\n我是llama3.1"
                    return final_answer
                else:
                    return "抱歉，無法從選定的文件中獲取答案"
                    
            except Exception as e:
                logger.error(f"RAG 處理過程中發生錯誤: {str(e)}")
                return f"處理文件時發生錯誤：{str(e)}"
        
        # 如果沒有選中文件，使用普通 LLM 回答
        else:
            if model_type == 'openai':
                # 使用 OpenAI API
                client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "你是一個有幫助的助手。請用繁體中文回答。"},
                        {"role": "user", "content": message}
                    ]
                )
                return response.choices[0].message.content
            else:
                # 使用 Ollama API
                # 定義 Ollama 的提示
                prompt = f"""請用繁體中文回答以下問題，並在回答的結尾說"我是llama3.1"：

問題：{message}"""

                response = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    return response.json()['response']
                else:
                    raise Exception(f"Ollama API 錯誤: {response.text}")
                
    except Exception as e:
        logger.error(f"處理請求時發生錯誤：{str(e)}")
        return f"發生錯誤：{str(e)}"

def ensure_database_exists():
    """確保資料庫存在，如果不存在則創建"""
    try:
        # 連接到默認的 postgres 數據庫
        engine = create_engine("postgresql+psycopg2://postgres:test@localhost:5432/postgres")
        with engine.connect() as connection:
            # 設置自動提交
            connection.execute(text("COMMIT"))
            # 檢查 vector_db 是否存在
            result = connection.execute(text(
                "SELECT 1 FROM pg_database WHERE datname = 'vector_db'"
            )).fetchone()
            
            if not result:
                # 創建新的數據庫
                connection.execute(text("COMMIT"))
                connection.execute(text("CREATE DATABASE vector_db"))
                logger.info("已創建 vector_db 數據庫")
                
                # 連接到新數據庫並創建 pgvector 擴展
                vector_engine = create_engine(CONNECTION_STRING)
                with vector_engine.connect() as vector_conn:
                    # 創建 vector 擴展
                    vector_conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    # 提交更改
                    vector_conn.execute(text("COMMIT"))
                    logger.info("已創建 vector_db 中創建 vector 擴展")
    except Exception as e:
        logger.error(f"確保數據庫存在時發生錯誤: {str(e)}")
        raise

def ensure_pgvector_setup():
    """確保pgvector擴展和必要的表都已創建"""
    try:
        engine = create_engine(CONNECTION_STRING)
        with engine.connect() as connection:
            # 創建 vector 擴展
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.commit()
            
            # 創建必要的表
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    collection_id TEXT,
                    embedding vector(1536),
                    document TEXT,
                    cmetadata JSONB,
                    custom_id TEXT
                );
            """))
            connection.commit()
            
            # 創建索引
            connection.execute(text("""
                CREATE INDEX IF NOT EXISTS langchain_pg_embedding_collection_id_idx 
                ON langchain_pg_embedding (collection_id);
            """))
            connection.commit()
            
            logger.info("PGVector 設置完成")
            
    except Exception as e:
        logger.error(f"設置 PGVector 時發生錯誤: {str(e)}")
        raise

def process_file_for_embedding(file_path, file_type):
    """處理文件並創建 embeddings"""
    try:
        # 確保 PGVector 設置完成
        ensure_pgvector_setup()
        
        # 載入文件
        logger.info(f"開始處理文件: {file_path}")
        if file_type == 'txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_type == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_type == 'md':
            loader = UnstructuredMarkdownLoader(file_path)
        elif file_type in ['doc', 'docx']:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            return False, "不支援的文件類型"

        # 載入文件內容
        documents = loader.load()
        logger.info(f"成功載入文件，共 {len(documents)} 個文檔")
        
        # 分割文本，使用與 ipynb 相同的參數
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 每個塊的大小
            chunk_overlap=80,  # 塊之間的重疊部分
            length_function=len,  # 使用字符長度
            is_separator_regex=False  # 不使用���則表達式為分隔符
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"文本分割完成，共 {len(splits)} 個片段")
        
        try:
            # 創建 embeddings
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            
            # 使用固定的 UUID 作為 collection_name
            collection_name = "f0d144a4-3d46-496e-b5b2-2d7684b21274"
            
            logger.info(f"正在創建向量存儲，collection名稱: {collection_name}")

            # 使用 PGVector 存儲向量
            try:
                # 先刪除已存在的collection
                engine = create_engine(CONNECTION_STRING)
                with engine.connect() as connection:
                    connection.execute(text("""
                        DELETE FROM langchain_pg_embedding 
                        WHERE collection_id = cast(:collection_id as uuid)
                    """), {"collection_id": collection_name})
                    connection.commit()
                    logger.info(f"已刪除舊的collection: {collection_name}")
            except Exception as e:
                logger.warning(f"刪除 collection 時發生錯誤: {str(e)}")

            # 創建新的向量存儲
            vector_store = PGVector.from_documents(
                documents=splits,
                embedding=embeddings,
                collection_name=collection_name,
                connection_string=CONNECTION_STRING
            )
            
            logger.info(f"向量存儲創建成功，collection: {collection_name}")
            logger.info(f"已處理 {len(splits)} 個文本片段")
            
            return True, collection_name
            
        except Exception as e:
            logger.error(f"創建 embeddings 失敗: {str(e)}")
            return False, str(e)
            
    except Exception as e:
        logger.error(f"處理文件時發生錯誤: {str(e)}")
        return False, str(e)

def get_uploads_files():
    """獲取上傳的文件列表"""
    try:
        sftp, transport = get_sftp_connection()
        if not sftp:
            return []
        
        try:
            # 構建完整的 uploads 路徑
            uploads_path = os.path.join(REMOTE_PATH, 'uploads')
            
            # 確保路徑存在
            try:
                sftp.stat(uploads_path)
            except FileNotFoundError:
                return []
            
            # 獲取文件列
            files = []
            for entry in sftp.listdir_attr(uploads_path):
                if stat.S_ISREG(entry.st_mode):  # 只處理常規文件
                    # 將時間戳轉換為本地時間
                    modified_time = datetime.fromtimestamp(entry.st_mtime)
                    files.append({
                        'name': entry.filename,
                        'modified': modified_time.strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            return sorted(files, key=lambda x: x['modified'], reverse=True)
            
        finally:
            sftp.close()
            transport.close()
            
    except Exception as e:
        logger.error(f"獲取上傳文件列表時發生錯誤: {str(e)}")
        return []

@app.route('/', methods=['GET', 'POST'])
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    try:
        # 檢查是否是新的session
        if 'initialized' not in session:
            session['messages'] = []
            session['current_model'] = 'ollama'
            session['initialized'] = True
            session.modified = True
            
        # 獲取上傳的文件列表
        uploaded_files = get_uploads_files()
        
        if request.method == 'POST':
            user_message = request.form.get('message')
            selected_model = request.form.get('model', 'ollama')
            use_knowledge_base = request.form.get('use_knowledge_base') == 'true'
            
            # 確保正確解析選中的文件
            try:
                selected_files = request.form.getlist('selected_files[]')  # 從表單中獲取選中的文件列表
                logger.info(f"選中的文件: {selected_files}")
            except Exception as e:
                selected_files = []
                logger.warning(f"解析選中文件時發生錯誤: {str(e)}")
            
            if user_message:
                session['messages'].append({
                    'type': 'user',
                    'content': user_message
                })
                
                # 獲取 AI 回應
                ai_response = get_ai_response(
                    user_message,
                    selected_model,
                    selected_files,  # 傳遞選中的文件
                    use_knowledge_base
                )
                
                session['messages'].append({
                    'type': 'ai',
                    'content': ai_response
                })
                
                session.modified = True
                
                return jsonify({
                    'response': ai_response
                })
            
            return jsonify({
                'error': '請輸入消息'
            })
                
    except Exception as e:
        logger.error(f"處理聊天請求時發生錯誤: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500
    
    return render_template('index.html',
                         messages=session.get('messages', []),
                         current_model='ollama',
                         uploaded_files=uploaded_files,
                         selected_files=session.get('selected_files', []))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            flash('沒有文件')
            return redirect(request.url)
        
        files = request.files.getlist('files[]')
        
        if not files or all(file.filename == '' for file in files):
            flash('未選擇文件')
            return redirect(request.url)
        
        success_count = 0
        error_count = 0
        processed_collections = []
        
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    # 使用原始文件名保存
                    filename = secure_filename(file.filename)
                    
                    # 創建臨時文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                        file.save(temp_file.name)
                        logger.info(f"已保存臨時文件: {temp_file.name}")
                        
                        # 獲取文件類型
                        file_type = filename.rsplit('.', 1)[1].lower()
                        
                        # 重新定位文件指到開始位置
                        file.seek(0)
                        
                        # 上傳到遠端時使原始文件名
                        date_path = 'uploads'  # 直接使用uploads目錄
                        remote_save_path = os.path.join(REMOTE_PATH, date_path)
                        
                        success, message = save_file_to_remote(file, remote_save_path, filename)
                        if not success:
                            raise Exception(f"遠端保存失敗：{message}")
                        
                        # 處理文件並創建 embeddings
                        success, collection_name = process_file_for_embedding(temp_file.name, file_type)
                        
                        if success:
                            success_count += 1
                            processed_collections.append(collection_name)
                            logger.info(f"文件 {filename} 已成功處理")
                        else:
                            raise Exception(f"創建 embeddings 失敗: {collection_name}")
                        
                except Exception as e:
                    error_count += 1
                    logger.error(f"處理文件 {file.filename} 時發生錯誤：{str(e)}")
                    flash(f'處理文件 {file.filename} 時發生錯誤：{str(e)}')
                finally:
                    if 'temp_file' in locals():
                        os.unlink(temp_file.name)
            else:
                error_count += 1
                flash(f'文件 {file.filename} 類型不允許')
        
        if success_count > 0:
            flash(f'成功處理 {success_count} 個文件')
        if error_count > 0:
            flash(f'有 {error_count} 個文件處理失敗')
        
        return redirect(url_for('chat'))
            
    return render_template('upload.html')

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    """清除聊天記錄"""
    if 'messages' in session:
        session['messages'] = []
        session.modified = True
    return redirect(url_for('chat'))

@app.route('/clear_collections', methods=['POST'])
def clear_collections():
    """清除所有集合"""
    try:
        engine = create_engine(CONNECTION_STRING)
        with engine.connect() as connection:
            connection.execute(text("DELETE FROM langchain_pg_embedding"))
            connection.execute(text("DELETE FROM langchain_pg_collection"))
            connection.execute(text("DELETE FROM langchain_collections"))
            connection.commit()
        flash('已清除所有集合')
    except Exception as e:
        flash(f'清除集合時發生錯誤: {str(e)}')
    return redirect(url_for('chat'))

@app.route('/clear_uploads', methods=['POST'])
def clear_uploads():
    """清除所有上傳的檔案和相關的資料庫記錄"""
    try:
        # 清除資料庫中的記錄
        engine = create_engine(CONNECTION_STRING)
        with engine.connect() as connection:
            # 清除有 langchain 相關的表
            connection.execute(text("DELETE FROM langchain_pg_embedding"))
            connection.execute(text("DELETE FROM langchain_pg_collection"))
            connection.execute(text("DELETE FROM langchain_collections"))
            
            connection.commit()
            logger.info("已清除所有料庫記錄")

        # 清除遠端檔案
        sftp, transport = get_sftp_connection()
        if not sftp:
            flash('無法連接到遠端伺服器')
            return redirect(url_for('chat'))
        
        try:
            uploads_path = os.path.join(REMOTE_PATH, 'uploads')
            logger.info(f"準備清除目錄: {uploads_path}")
            
            try:
                sftp.stat(uploads_path)
            except FileNotFoundError:
                flash('uploads目錄不存在')
                return redirect(url_for('chat'))
            
            try:
                files = sftp.listdir(uploads_path)
                for filename in files:
                    if filename not in ['.', '..']:
                        file_path = os.path.join(uploads_path, filename)
                        try:
                            sftp.remove(file_path)
                            logger.info(f"已刪除檔案: {file_path}")
                        except Exception as e:
                            logger.error(f"刪除檔案 {file_path} 時發生錯誤: {str(e)}")
                            continue
            except Exception as e:
                logger.error(f"列出檔案時發生錯誤: {str(e)}")
                flash(f'列出檔案時發生錯誤: {str(e)}')
                return redirect(url_for('chat'))
            
            flash('已清除所有上傳的檔案和相關資料庫記錄')
            
        except Exception as e:
            flash(f'清除檔案時發生錯誤: {str(e)}')
            logger.error(f"清除上傳檔案時發生錯誤: {str(e)}")
        finally:
            sftp.close()
            transport.close()
            
    except Exception as e:
        flash(f'操作時發生錯誤: {str(e)}')
        logger.error(f"清除操作時發生錯誤: {str(e)}")
    
    return redirect(url_for('chat'))

if __name__ == '__main__':
    try:
        logger.info("啟動 Flask 應用序...")
        # 嘗試使用其他端口，例如 5000
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        logger.error(f"啟動應用程序時發生錯誤: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())