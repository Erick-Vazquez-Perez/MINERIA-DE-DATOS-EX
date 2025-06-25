# app.py - Aplicación de Minería de Datos Educativa (Versión Profesional)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import re

# Configuración de la página
st.set_page_config(
    page_title="Minería de Datos Educativa",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS profesional
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
        border-bottom: 3px solid #3498db;
        padding-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #34495e;
        border-bottom: 2px solid #bdc3c7;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .info-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    .metric-label {
        color: #6c757d;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .stButton > button {
        background: linear-gradient(90deg, #3498db 0%, #2980b9 100%);
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
    }
    .qa-question {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        border-left: 3px solid #007bff;
        margin: 0.5rem 0;
    }
    .qa-response {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 6px;
        border-left: 3px solid #28a745;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DataMiningEducationalApp:
    def __init__(self):
        # Modelos de ML
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Naive Bayes': GaussianNB(),
            'SVM': SVC(random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Neural Network': MLPClassifier(random_state=42, max_iter=500)
        }
        
        self.scaler = StandardScaler()
        
        # Base de conocimiento de minería de datos
        self.knowledge_base = {
            "minería de datos": {
                "definition": "Proceso de descubrir patrones en grandes conjuntos de datos mediante métodos de inteligencia artificial, machine learning y estadística. Su propósito es la obtención de información no trivial para predicción y toma de decisiones.",
                "keywords": ["patterns", "data", "discovery", "algorithms", "predicción"],
                "related": ["machine learning", "data warehouse", "analytics", "business intelligence"]
            },
            "data warehouse": {
                "definition": "Sistema de almacenamiento centralizado que integra datos de múltiples fuentes para análisis y toma de decisiones. Caracterizado por ser orientado a temas, variante en el tiempo, no volátil e integrado.",
                "keywords": ["storage", "integration", "analysis", "olap", "centralizado"],
                "related": ["etl", "olap", "business intelligence", "molap", "rolap"]
            },
            "olap": {
                "definition": "On-Line Analytical Processing. Tecnología que permite análisis multidimensional rápido de datos empresariales mediante cubos de datos y jerarquías dimensionales.",
                "keywords": ["multidimensional", "analysis", "cubes", "aggregation", "online"],
                "related": ["molap", "rolap", "holap", "data warehouse", "cubos"]
            },
            "molap": {
                "definition": "Multidimensional OLAP. Usa bases de datos multidimensionales propietarias para almacenar información multidimensionalmente, optimizando el acceso y análisis de datos.",
                "keywords": ["multidimensional", "proprietary", "storage", "optimization"],
                "related": ["olap", "rolap", "holap", "cubos de datos"]
            },
            "rolap": {
                "definition": "Relational OLAP. Utiliza sistemas de gestión de bases de datos relacionales (RDBMS) para OLAP, empleando esquemas de estrella y copo de nieve.",
                "keywords": ["relational", "rdbms", "star schema", "snowflake"],
                "related": ["olap", "molap", "esquemas", "bases relacionales"]
            },
            "holap": {
                "definition": "Hybrid OLAP. Combina arquitecturas ROLAP y MOLAP para brindar desempeño superior y escalabilidad, manteniendo detalles en bases relacionales y agregaciones en almacén multidimensional.",
                "keywords": ["hybrid", "scalability", "performance", "combination"],
                "related": ["molap", "rolap", "olap", "arquitectura híbrida"]
            },
            "etl": {
                "definition": "Extract, Transform, Load. Proceso de extracción de datos desde fuentes, transformación para limpieza y estructuración, y carga en el sistema destino como data warehouse.",
                "keywords": ["extraction", "transformation", "loading", "integration", "proceso"],
                "related": ["data warehouse", "integration", "data pipeline", "procesamiento"]
            },
            "machine learning": {
                "definition": "Rama de la inteligencia artificial que permite a las máquinas aprender patrones de los datos sin programación explícita, enfocándose en predicción mediante algoritmos.",
                "keywords": ["learning", "algorithms", "prediction", "models", "artificial intelligence"],
                "related": ["clasificación", "regresión", "clustering", "neural networks", "minería de datos"]
            },
            "clasificación": {
                "definition": "Técnica de machine learning supervisado que predice categorías o clases de nuevos datos basándose en patrones aprendidos de datos etiquetados.",
                "keywords": ["supervised", "categories", "prediction", "labels", "clases"],
                "related": ["random forest", "svm", "naive bayes", "decision tree", "machine learning"]
            },
            "clustering": {
                "definition": "Técnica de aprendizaje no supervisado que agrupa datos similares sin etiquetas previas, identificando estructuras ocultas en los datos.",
                "keywords": ["unsupervised", "groups", "similarity", "patterns", "agrupación"],
                "related": ["k-means", "hierarchical", "dbscan", "machine learning"]
            },
            "business intelligence": {
                "definition": "Conjunto de herramientas y procesos para convertir datos en información relevante mediante visualización, análogo a estadística descriptiva para mejor entendimiento empresarial.",
                "keywords": ["visualization", "dashboards", "reports", "analytics", "empresarial"],
                "related": ["data warehouse", "olap", "analytics", "reportes"]
            },
            "big data": {
                "definition": "Información estructurada, no estructurada y semiestructurada que no puede procesarse con herramientas tradicionales, caracterizada por volumen, velocidad y variedad.",
                "keywords": ["volume", "velocity", "variety", "terabytes", "processing"],
                "related": ["hadoop", "spark", "nosql", "distributed computing"]
            }
        }
        
        # Inicializar session state
        if 'qa_history' not in st.session_state:
            st.session_state['qa_history'] = []
        if 'current_dataset' not in st.session_state:
            st.session_state['current_dataset'] = None
    
    def load_built_in_dataset(self, dataset_name):
        """Carga datasets integrados para demostración"""
        if dataset_name == "Iris":
            from sklearn.datasets import load_iris
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
            return df, "Dataset de clasificación de especies de flores Iris"
            
        elif dataset_name == "Wine Quality":
            from sklearn.datasets import load_wine
            data = load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['wine_class'] = df['target'].map({0: 'class_0', 1: 'class_1', 2: 'class_2'})
            return df, "Dataset de clasificación de calidad de vinos"
            
        elif dataset_name == "Breast Cancer":
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['diagnosis'] = df['target'].map({0: 'malignant', 1: 'benign'})
            return df, "Dataset de diagnóstico de cáncer de mama"
    
    def process_question(self, question):
        """Procesa pregunta y genera respuesta basada en conocimiento integrado"""
        question_clean = question.lower().strip()
        
        # Buscar coincidencias en base de conocimiento
        best_match = self.find_best_match(question_clean)
        
        if best_match:
            response = self.generate_response(best_match, question_clean)
        else:
            response = self.generate_general_response(question_clean)
        
        # Guardar en historial
        st.session_state['qa_history'].append({
            'question': question,
            'response': response,
            'timestamp': datetime.now()
        })
        
        return response
    
    def find_best_match(self, question):
        """Encuentra mejor coincidencia en base de conocimiento"""
        best_score = 0
        best_match = None
        
        for concept, data in self.knowledge_base.items():
            score = 0
            
            # Coincidencia directa del concepto
            if concept in question:
                score += 10
            
            # Palabras clave
            for keyword in data['keywords']:
                if keyword in question:
                    score += 2
            
            # Conceptos relacionados
            for related in data['related']:
                if related in question:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = concept
        
        return best_match if best_score > 0 else None
    
    def generate_response(self, concept, question):
        """Genera respuesta basada en concepto encontrado"""
        data = self.knowledge_base[concept]
        
        response = f"**{concept.title()}**\n\n"
        response += f"{data['definition']}\n\n"
        
        if data['related']:
            response += f"**Conceptos relacionados:** {', '.join(data['related'])}"
        
        return response
    
    def generate_general_response(self, question):
        """Genera respuesta general cuando no encuentra coincidencia específica"""
        # Identificar categoría general
        if any(word in question for word in ['algoritmo', 'modelo', 'clasificacion', 'regresion']):
            return """**Machine Learning**

Los algoritmos de machine learning se dividen en:
- **Supervisados**: Clasificación y regresión con datos etiquetados
- **No supervisados**: Clustering y reducción de dimensionalidad
- **Por refuerzo**: Aprendizaje mediante recompensas

**Algoritmos principales**: Random Forest, SVM, Naive Bayes, Redes Neuronales"""
        
        elif any(word in question for word in ['datos', 'informacion', 'base']):
            return """**Gestión de Datos**

La gestión efectiva de datos involucra:
- **Calidad**: Integridad, consistencia y completitud
- **Almacenamiento**: Bases de datos, data warehouses
- **Procesamiento**: ETL, transformaciones
- **Análisis**: OLAP, minería de datos"""
        
        else:
            return """**Información General**

Esta aplicación cubre conceptos fundamentales de:
- Minería de datos y sus procesos
- Data warehousing y arquitecturas OLAP
- Machine learning y algoritmos de clasificación
- Business intelligence y analytics

Para consultas específicas, utiliza términos como 'OLAP', 'minería de datos', 'clasificación', etc."""
    
    def train_models(self, X, y, test_size=0.2, cv_folds=5):
        """Entrena modelos con validación cruzada"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Escalar datos
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            try:
                # Entrenamiento
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Validación cruzada
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)
                
                results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'true_values': y_test,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
            except Exception as e:
                st.warning(f"Error entrenando {name}: {str(e)}")
                results[name] = None
        
        return results, X_test_scaled, y_test

def main():
    # Título principal
    st.markdown('<h1 class="main-header">Sistema de Minería de Datos Educativa</h1>', unsafe_allow_html=True)
    
    # Inicializar aplicación
    app = DataMiningEducationalApp()
    
    # Sidebar
    st.sidebar.title("Panel de Control")
    st.sidebar.markdown("---")
    
    # Navegación
    page = st.sidebar.selectbox(
        "Seleccionar sección:",
        ["Dashboard Principal", "Sistema de Consultas", "Laboratorio de Machine Learning", "Análisis de Resultados"]
    )
    
    # Estado del sistema en sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Estado del Sistema")
    
    if st.session_state.get('current_dataset') is not None:
        dataset_info = st.session_state['current_dataset']
        st.sidebar.success(f"Dataset activo: {dataset_info['name']}")
        st.sidebar.info(f"Filas: {dataset_info['data'].shape[0]}")
    else:
        st.sidebar.warning("Sin dataset activo")
    
    if st.session_state.get('ml_results'):
        results = st.session_state['ml_results']
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_model = max(valid_results.items(), key=lambda x: x[1]['accuracy'])
            st.sidebar.info(f"Mejor modelo: {best_model[0]} ({best_model[1]['accuracy']:.3f})")
    
    # Navegación a páginas
    if page == "Dashboard Principal":
        show_dashboard(app)
    elif page == "Sistema de Consultas":
        show_qa_system(app)
    elif page == "Laboratorio de Machine Learning":
        show_ml_lab(app)
    elif page == "Análisis de Resultados":
        show_results_analysis(app)

def show_dashboard(app):
    """Dashboard principal"""
    st.markdown('<h2 class="section-header">Dashboard Principal</h2>', unsafe_allow_html=True)
    
    # Métricas del sistema
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">7</div>
            <div class="metric-label">Algoritmos ML</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        qa_count = len(st.session_state.get('qa_history', []))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{qa_count}</div>
            <div class="metric-label">Consultas Realizadas</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        knowledge_items = len(app.knowledge_base)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{knowledge_items}</div>
            <div class="metric-label">Conceptos en Base</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        model_count = len(st.session_state.get('ml_results', {}))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{model_count}</div>
            <div class="metric-label">Modelos Entrenados</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Información del sistema
    st.markdown('<h3 class="section-header">Información del Sistema</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>Objetivos del Sistema</h4>
        <ul>
            <li>Procesamiento de consultas educativas sobre minería de datos</li>
            <li>Clasificación automática de preguntas por tema y tipo</li>
            <li>Entrenamiento y evaluación de modelos de machine learning</li>
            <li>Análisis comparativo de algoritmos de clasificación</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>Funcionalidades Disponibles</h4>
        <ul>
            <li>Sistema de consultas inteligente con base de conocimiento</li>
            <li>Laboratorio ML con 7 algoritmos de clasificación</li>
            <li>Evaluación avanzada con métricas especializadas</li>
            <li>Análisis comparativo y recomendaciones automáticas</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Guía de uso
    st.markdown('<h3 class="section-header">Guía de Uso</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>Flujo de Trabajo Recomendado</h4>
    <ol>
        <li><strong>Sistema de Consultas:</strong> Realiza preguntas sobre conceptos de minería de datos</li>
        <li><strong>Laboratorio ML:</strong> Carga un dataset y entrena múltiples modelos</li>
        <li><strong>Análisis de Resultados:</strong> Evalúa y compara el rendimiento de los modelos</li>
        <li><strong>Interpretación:</strong> Utiliza las recomendaciones para seleccionar el mejor modelo</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

def show_qa_system(app):
    """Sistema de consultas inteligente"""
    st.markdown('<h2 class="section-header">Sistema de Consultas Inteligente</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>Asistente de Minería de Datos</h4>
    <p>Sistema de preguntas y respuestas basado en conocimiento integrado sobre minería de datos, 
    data warehousing, OLAP y machine learning. Proporciona respuestas contextuales y precisas.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Interfaz de consulta
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_question = st.text_input(
            "Escribe tu consulta:",
            placeholder="Ejemplo: ¿Qué es OLAP? ¿Cuáles son los tipos de machine learning?"
        )
    
    with col2:
        ask_button = st.button("Consultar", type="primary")
    
    # Procesar consulta
    if ask_button and user_question:
        with st.spinner("Procesando consulta..."):
            response = app.process_question(user_question)
            
            # Mostrar respuesta
            st.markdown(f"""
            <div class="qa-question">
            <strong>Pregunta:</strong> {user_question}
            </div>
            <div class="qa-response">
            <strong>Respuesta:</strong><br>
            {response}
            </div>
            """, unsafe_allow_html=True)
    
    # Consultas sugeridas
    st.markdown('<h3 class="section-header">Consultas Sugeridas</h3>', unsafe_allow_html=True)
    
    suggested_questions = [
        "¿Qué es la minería de datos?",
        "¿Cuáles son los tipos de OLAP?",
        "¿Qué diferencia hay entre ROLAP y MOLAP?",
        "¿Qué es un data warehouse?",
        "¿Cuáles son los algoritmos de clasificación más comunes?",
        "¿Qué es el proceso ETL?",
        "¿Cuándo usar Random Forest vs SVM?",
        "¿Qué es clustering en machine learning?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(suggested_questions):
        with cols[i % 2]:
            if st.button(question, key=f"suggest_{i}"):
                response = app.process_question(question)
                st.markdown(f"""
                <div class="qa-question">
                <strong>Pregunta:</strong> {question}
                </div>
                <div class="qa-response">
                <strong>Respuesta:</strong><br>
                {response}
                </div>
                """, unsafe_allow_html=True)
    
    # Historial de consultas
    if st.session_state.get('qa_history'):
        st.markdown('<h3 class="section-header">Historial de Consultas</h3>', unsafe_allow_html=True)
        
        for i, qa in enumerate(reversed(st.session_state['qa_history'][-5:])):
            with st.expander(f"Consulta {len(st.session_state['qa_history'])-i}: {qa['question'][:50]}..."):
                st.write(f"**Pregunta:** {qa['question']}")
                st.write(f"**Respuesta:** {qa['response']}")
                st.write(f"**Fecha:** {qa['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

def show_ml_lab(app):
    """Laboratorio de Machine Learning"""
    st.markdown('<h2 class="section-header">Laboratorio de Machine Learning</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>Entrenamiento y Evaluación de Modelos</h4>
    <p>Laboratorio para entrenar múltiples algoritmos de clasificación con validación cruzada 
    y comparación de rendimiento. Utiliza datasets integrados para demostración.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Selección de dataset
    st.markdown('<h3 class="section-header">Selección de Dataset</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_name = st.selectbox(
            "Dataset para entrenamiento:",
            ["Iris", "Wine Quality", "Breast Cancer"]
        )
    
    with col2:
        if st.button("Cargar Dataset", type="primary"):
            df, description = app.load_built_in_dataset(dataset_name)
            st.session_state['current_dataset'] = {
                'name': dataset_name,
                'data': df,
                'description': description
            }
            st.success(f"Dataset {dataset_name} cargado exitosamente")
            st.rerun()
    
    # Mostrar información del dataset
    if st.session_state.get('current_dataset'):
        dataset_info = st.session_state['current_dataset']
        df = dataset_info['data']
        
        st.markdown('<h3 class="section-header">Información del Dataset</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Filas", df.shape[0])
        with col2:
            st.metric("Columnas", df.shape[1])
        with col3:
            st.metric("Valores Nulos", df.isnull().sum().sum())
        with col4:
            numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
            st.metric("Cols. Numéricas", numeric_cols)
        
        st.info(f"**Descripción:** {dataset_info['description']}")
        
        # Vista previa
        with st.expander("Vista previa de datos"):
            st.dataframe(df.head())
        
        # Configuración del entrenamiento
        st.markdown('<h3 class="section-header">Configuración del Entrenamiento</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_column = st.selectbox(
                "Columna objetivo:",
                [col for col in df.columns if col in ['target', 'species', 'wine_class', 'diagnosis']]
            )
        
        with col2:
            test_size = st.slider("Tamaño conjunto de prueba:", 0.1, 0.5, 0.2, 0.05)
        
        with col3:
            cv_folds = st.slider("Pliegues validación cruzada:", 3, 10, 5)
        
        # Entrenamiento
        if st.button("Entrenar Modelos", type="primary"):
            with st.spinner("Entrenando modelos..."):
                # Preparar datos
                X = df.drop([target_column], axis=1)
                y = df[target_column]
                
                # Codificar variables categóricas
                for col in X.select_dtypes(include=['object']).columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                
                # Entrenar modelos
                results, X_test, y_test = app.train_models(X, y, test_size, cv_folds)
                
                # Guardar resultados
                st.session_state['ml_results'] = results
                st.session_state['test_data'] = (X_test, y_test)
                
                st.success("Entrenamiento completado exitosamente")
                
                # Resumen rápido
                st.markdown('<h4>Resumen de Resultados</h4>', unsafe_allow_html=True)
                
                valid_results = {k: v for k, v in results.items() if v is not None}
                if valid_results:
                    summary_data = []
                    for name, result in valid_results.items():
                        summary_data.append({
                            'Modelo': name,
                            'Precisión': f"{result['accuracy']:.4f}",
                            'CV Media': f"{result['cv_mean']:.4f}",
                            'CV Std': f"{result['cv_std']:.4f}"
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Gráfico de comparación
                    accuracies = [r['accuracy'] for r in valid_results.values()]
                    model_names = list(valid_results.keys())
                    
                    fig = px.bar(
                        x=model_names, 
                        y=accuracies,
                        title="Comparación de Precisión por Modelo",
                        labels={'x': 'Modelo', 'y': 'Precisión'}
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.info("Dirígete a 'Análisis de Resultados' para evaluación detallada")

def show_results_analysis(app):
    """Análisis detallado de resultados"""
    st.markdown('<h2 class="section-header">Análisis de Resultados</h2>', unsafe_allow_html=True)
    
    if 'ml_results' not in st.session_state:
        st.warning("Primero entrena modelos en el Laboratorio de Machine Learning")
        return
    
    results = st.session_state['ml_results']
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        st.error("No hay resultados válidos de entrenamiento")
        return
    
    st.markdown("""
    <div class="info-box">
    <h4>Evaluación Comprehensiva de Modelos</h4>
    <p>Análisis detallado del rendimiento de los modelos entrenados con métricas especializadas, 
    matrices de confusión y recomendaciones automáticas.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs para diferentes análisis
    tab1, tab2, tab3, tab4 = st.tabs(["Métricas Generales", "Análisis Detallado", "Comparaciones", "Recomendaciones"])
    
    with tab1:
        st.subheader("Métricas de Rendimiento")
        
        # Tabla de métricas
        metrics_data = []
        for name, result in valid_results.items():
            metrics_data.append({
                'Modelo': name,
                'Precisión': result['accuracy'],
                'CV Media': result['cv_mean'],
                'CV Std': result['cv_std'],
                'Estabilidad': 'Alta' if result['cv_std'] < 0.05 else 'Media' if result['cv_std'] < 0.1 else 'Baja'
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df = metrics_df.sort_values('Precisión', ascending=False)
        
        # Mostrar tabla con formato
        st.dataframe(
            metrics_df.style.format({
                'Precisión': '{:.4f}',
                'CV Media': '{:.4f}',
                'CV Std': '{:.4f}'
            }).background_gradient(subset=['Precisión'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        # Ranking de modelos
        st.subheader("Ranking de Modelos")
        ranking_data = []
        for i, (_, row) in enumerate(metrics_df.iterrows()):
            ranking_data.append({
                'Posición': i + 1,
                'Modelo': row['Modelo'],
                'Puntuación': row['Precisión'],
                'Evaluación': 'Excelente' if row['Precisión'] >= 0.95 else 'Muy Bueno' if row['Precisión'] >= 0.90 else 'Bueno' if row['Precisión'] >= 0.80 else 'Regular'
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        st.dataframe(ranking_df, use_container_width=True)
    
    with tab2:
        st.subheader("Análisis por Modelo")
        
        # Selector de modelo
        selected_model = st.selectbox(
            "Seleccionar modelo para análisis:",
            list(valid_results.keys())
        )
        
        if selected_model:
            result = valid_results[selected_model]
            y_true = result['true_values']
            y_pred = result['predictions']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Matriz de confusión
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'Matriz de Confusión - {selected_model}')
                ax.set_xlabel('Predicciones')
                ax.set_ylabel('Valores Reales')
                st.pyplot(fig)
            
            with col2:
                # Reporte de clasificación
                st.write("**Reporte de Clasificación:**")
                report = classification_report(y_true, y_pred, output_dict=True)
                
                # Convertir a DataFrame
                report_df = pd.DataFrame(report).transpose()
                report_df = report_df.drop(['accuracy'], errors='ignore')
                st.dataframe(report_df.round(4))
            
            # Métricas específicas del modelo
            st.subheader(f"Métricas Detalladas - {selected_model}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Precisión", f"{result['accuracy']:.4f}")
            with col2:
                st.metric("CV Media", f"{result['cv_mean']:.4f}")
            with col3:
                st.metric("CV Std", f"{result['cv_std']:.4f}")
            with col4:
                cv_stability = "Alta" if result['cv_std'] < 0.05 else "Media" if result['cv_std'] < 0.1 else "Baja"
                st.metric("Estabilidad", cv_stability)
    
    with tab3:
        st.subheader("Comparaciones Visuales")
        
        # Gráfico de barras comparativo
        model_names = list(valid_results.keys())
        accuracies = [valid_results[name]['accuracy'] for name in model_names]
        cv_means = [valid_results[name]['cv_mean'] for name in model_names]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Precisión en Test', 'Validación Cruzada Media')
        )
        
        fig.add_trace(
            go.Bar(x=model_names, y=accuracies, name='Precisión Test'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=model_names, y=cv_means, name='CV Media'),
            row=1, col=2
        )
        
        fig.update_layout(title_text="Comparación de Rendimiento", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de dispersión: Precisión vs Estabilidad
        st.subheader("Análisis de Precisión vs Estabilidad")
        
        scatter_data = []
        for name, result in valid_results.items():
            scatter_data.append({
                'Modelo': name,
                'Precisión': result['accuracy'],
                'Estabilidad': 1 / (result['cv_std'] + 0.001)  # Inverso de std para que mayor sea mejor
            })
        
        scatter_df = pd.DataFrame(scatter_data)
        fig = px.scatter(
            scatter_df, 
            x='Estabilidad', 
            y='Precisión',
            text='Modelo',
            title="Precisión vs Estabilidad de Modelos"
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de validación cruzada
        st.subheader("Distribución de Scores de Validación Cruzada")
        
        cv_data = []
        for model_name, result in valid_results.items():
            for score in result['cv_scores']:
                cv_data.append({
                    'Modelo': model_name,
                    'CV_Score': score
                })
        
        cv_df = pd.DataFrame(cv_data)
        fig = px.box(
            cv_df, 
            x='Modelo', 
            y='CV_Score',
            title="Distribución de Scores de Validación Cruzada"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Recomendaciones Automáticas")
        
        # Generar recomendaciones
        recommendations = generate_recommendations(valid_results)
        
        for i, rec in enumerate(recommendations):
            st.markdown(f"""
            <div class="info-box">
            <h4>Recomendación {i+1}</h4>
            <p>{rec}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Modelo recomendado
        st.subheader("Modelo Recomendado")
        
        # Calcular puntuación combinada (precisión + estabilidad)
        combined_scores = {}
        for name, result in valid_results.items():
            # Puntuación combinada: 70% precisión + 30% estabilidad (inverso de std)
            stability_score = max(0, 1 - result['cv_std'] * 10)  # Normalizar estabilidad
            combined_score = 0.7 * result['accuracy'] + 0.3 * stability_score
            combined_scores[name] = combined_score
        
        best_model = max(combined_scores.items(), key=lambda x: x[1])
        
        st.markdown(f"""
        <div class="success-box">
        <h4>Modelo Recomendado: {best_model[0]}</h4>
        <p><strong>Puntuación Combinada:</strong> {best_model[1]:.4f}</p>
        <p><strong>Precisión:</strong> {valid_results[best_model[0]]['accuracy']:.4f}</p>
        <p><strong>Estabilidad CV:</strong> {valid_results[best_model[0]]['cv_std']:.4f}</p>
        <p><strong>Justificación:</strong> Este modelo ofrece el mejor balance entre precisión y estabilidad según los criterios de evaluación.</p>
        </div>
        """, unsafe_allow_html=True)

def generate_recommendations(results):
    """Genera recomendaciones basadas en resultados"""
    recommendations = []
    
    # Mejor precisión
    best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
    recommendations.append(
        f"**Mejor Precisión:** {best_accuracy[0]} con {best_accuracy[1]['accuracy']:.4f}. "
        f"Recomendado para casos donde la precisión es prioritaria."
    )
    
    # Mejor estabilidad
    best_stability = min(results.items(), key=lambda x: x[1]['cv_std'])
    recommendations.append(
        f"**Mayor Estabilidad:** {best_stability[0]} con CV std de {best_stability[1]['cv_std']:.4f}. "
        f"Recomendado para entornos de producción donde la consistencia es clave."
    )
    
    # Análisis de rendimiento general
    avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
    
    if avg_accuracy > 0.95:
        recommendations.append(
            "**Rendimiento Excepcional:** Todos los modelos muestran alta precisión. "
            "El dataset parece bien estructurado y los algoritmos son apropiados."
        )
    elif avg_accuracy > 0.85:
        recommendations.append(
            "**Buen Rendimiento:** Los modelos muestran resultados satisfactorios. "
            "Considera optimización de hiperparámetros para mejorar aún más."
        )
    else:
        recommendations.append(
            "**Rendimiento Mejorable:** Los resultados sugieren necesidad de más datos "
            "o mejor preprocesamiento. Revisa la calidad del dataset."
        )
    
    # Recomendaciones específicas por modelo
    for model_name, result in results.items():
        if model_name == "Random Forest" and result['accuracy'] > 0.90:
            recommendations.append(
                "**Random Forest** muestra excelente rendimiento. Es robusto y maneja bien "
                "características no lineales. Recomendado para implementación en producción."
            )
        elif model_name == "SVM" and result['cv_std'] < 0.05:
            recommendations.append(
                "**SVM** demuestra alta estabilidad. Excelente para datasets con separación "
                "clara entre clases y cuando la interpretabilidad no es crítica."
            )
        elif model_name == "Logistic Regression" and result['accuracy'] > 0.85:
            recommendations.append(
                "**Logistic Regression** ofrece buen balance entre rendimiento e interpretabilidad. "
                "Ideal cuando necesitas explicar las decisiones del modelo."
            )
    
    return recommendations

if __name__ == "__main__":
    main()