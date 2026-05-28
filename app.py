import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Pink RAG Assistant",
    page_icon="💖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# ESTILOS CSS PERSONALIZADOS
# ─────────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Fondo */
.stApp {
    background: linear-gradient(135deg, #0f0f14, #1c1023);
    color: white;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161621;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Sidebar títulos */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ff85c2 !important;
    font-weight: 700 !important;
}

/* Sidebar textos */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #f5f5f7 !important;
}

/* Títulos */
h1 {
    color: #ff4fa3 !important;
    font-size: 3.2rem !important;
    font-weight: 700 !important;
    text-align: center;
}

h2, h3 {
    color: #ff85c2 !important;
}

/* Texto */
p, span, label {
    color: #f5f5f7 !important;
}

/* Cards */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 28px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    margin-bottom: 22px;
}

/* Inputs */
textarea,
input[type="text"],
input[type="password"] {
    background-color: #1f1f2e !important;
    border: 2px solid #ff4fa3 !important;
    border-radius: 14px !important;
    color: white !important;
    padding: 12px !important;
}

/* Text Area */
textarea {
    min-height: 120px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    padding: 18px;
    border-radius: 20px;
    border: 2px dashed #ff4fa3;
}

/* Botones */
.stButton > button {
    background: linear-gradient(135deg, #ff4fa3, #ff85c2) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.8rem 1.5rem !important;
    font-weight: 600 !important;
    transition: 0.3s ease !important;
    box-shadow: 0 4px 18px rgba(255,79,163,0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(255,79,163,0.45);
}

/* Alertas */
.stAlert {
    border-radius: 18px !important;
}

/* Imagen */
img {
    border-radius: 18px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.25);
}

/* Resultado respuesta */
.response-card {
    background: rgba(255,255,255,0.05);
    border-left: 5px solid #ff4fa3;
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.2);
    line-height: 1.8;
}

/* Métricas */
.metric-card {
    background: rgba(255,255,255,0.04);
    border-radius: 20px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Scroll */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #ff4fa3;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="glass-card">

<h1>💖 Pink RAG Assistant</h1>

<p style='text-align:center; font-size:18px; margin-top:10px;'>
Sistema de Generación Aumentada por Recuperación (RAG) para análisis inteligente de PDFs mediante IA.
</p>

</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INFO SUPERIOR
# ─────────────────────────────────────────────
col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown(f"""
    <div class="metric-card">
    <h3>🐍 Python</h3>
    <p style="font-size:20px;">{platform.python_version()}</p>
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    st.markdown("""
    <div class="metric-card">
    <h3>🤖 Modelo IA</h3>
    <p style="font-size:20px;">GPT + LangChain + FAISS</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LAYOUT PRINCIPAL
# ─────────────────────────────────────────────
col1, col2 = st.columns([1,1], gap="large")

# ─────────────────────────────────────────────
# COLUMNA IZQUIERDA
# ─────────────────────────────────────────────
with col1:

    st.markdown("""
    <div class="glass-card">

    <h3>🔑 Configuración OpenAI</h3>

    <p>
    Ingresa tu clave API para habilitar el análisis inteligente de documentos PDF.
    </p>

    </div>
    """, unsafe_allow_html=True)

    ke = st.text_input(
        'Ingresa tu Clave de OpenAI',
        type="password"
    )

    if ke:
        os.environ['OPENAI_API_KEY'] = ke
    else:
        st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

    st.markdown("""
    <div class="glass-card">

    <h3>📄 Subir Documento PDF</h3>

    <p>
    Carga un archivo PDF para comenzar el análisis mediante recuperación aumentada.
    </p>

    </div>
    """, unsafe_allow_html=True)

    pdf = st.file_uploader(
        "Carga el archivo PDF",
        type="pdf"
    )

# ─────────────────────────────────────────────
# COLUMNA DERECHA
# ─────────────────────────────────────────────
with col2:

    st.markdown("""
    <div class="glass-card">

    <h3>🧠 Capacidades del Sistema</h3>

    <p>
    El sistema divide el documento en fragmentos, genera embeddings semánticos y responde preguntas utilizando búsqueda contextual inteligente.
    </p>

    </div>
    """, unsafe_allow_html=True)

    try:
        image = Image.open('Chat_pdf.png')

        st.image(
            image,
            width=400
        )

    except Exception as e:
        st.warning(f"No se pudo cargar la imagen: {e}")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    st.markdown("## 💖 Pink RAG Assistant")

    st.markdown("""
    <div class="glass-card">

    Este asistente te ayudará a:
    
    • Analizar documentos PDF  
    • Buscar información específica  
    • Generar respuestas contextuales  
    • Utilizar IA con embeddings y recuperación semántica  

    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PROCESAMIENTO PDF
# ─────────────────────────────────────────────
if pdf is not None and ke:

    try:

        pdf_reader = PdfReader(pdf)

        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        st.info(f"Texto extraído: {len(text)} caracteres")

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        st.success(f"Documento dividido en {len(chunks)} fragmentos")

        embeddings = OpenAIEmbeddings()

        knowledge_base = FAISS.from_texts(
            chunks,
            embeddings
        )

        st.markdown("""
        <div class="glass-card">

        <h3>💬 Haz una pregunta</h3>

        <p>
        Consulta cualquier información contenida dentro del PDF cargado.
        </p>

        </div>
        """, unsafe_allow_html=True)

        user_question = st.text_area(
            " ",
            placeholder="Escribe tu pregunta aquí..."
        )

        if user_question:

            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI(
                temperature=0,
                model_name="gpt-4o-mini-2024-07-18"
            )

            chain = load_qa_chain(
                llm,
                chain_type="stuff"
            )

            response = chain.run(
                input_documents=docs,
                question=user_question
            )

            st.markdown("""
            <div class="response-card">

            <h3>📝 Respuesta Generada</h3>

            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="glass-card">

            {response}

            </div>
            """, unsafe_allow_html=True)

    except Exception as e:

        st.error(f"Error al procesar el PDF: {str(e)}")

        import traceback

        st.error(traceback.format_exc())

elif pdf is not None and not ke:

    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

else:

    st.info("📄 Por favor carga un archivo PDF para comenzar")
