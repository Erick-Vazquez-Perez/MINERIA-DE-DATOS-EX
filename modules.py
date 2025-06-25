# modules.py - Módulos adicionales para funcionalidades avanzadas
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
from datetime import datetime
import io
import base64

class ContentProcessor:
    """Clase para procesar contenido educativo (PPTX, PDF, Notebooks)"""
    
    def __init__(self):
        self.content_index = {}
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        
    def process_text_content(self, text, content_type="general"):
        """Procesa y almacena contenido de texto"""
        # Limpiar texto
        cleaned_text = self.clean_text(text)
        
        # Extraer conceptos clave
        concepts = self.extract_concepts(cleaned_text)
        
        # Almacenar en índice
        content_id = f"{content_type}_{datetime.now().timestamp()}"
        self.content_index[content_id] = {
            'text': cleaned_text,
            'concepts': concepts,
            'type': content_type,
            'timestamp': datetime.now(),
            'processed': True
        }
        
        return content_id
    
    def clean_text(self, text):
        """Limpia y normaliza texto"""
        # Remover caracteres especiales
        text = re.sub(r'[^\w\s]', ' ', text)
        # Convertir a minúsculas
        text = text.lower()
        # Remover espacios extra
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_concepts(self, text):
        """Extrae conceptos clave del texto"""
        # Lista de conceptos relacionados con minería de datos
        data_mining_concepts = [
            'minería de datos', 'data mining', 'machine learning', 'aprendizaje automático',
            'data warehouse', 'olap', 'oltp', 'etl', 'extracción', 'transformación',
            'algoritmo', 'clasificación', 'regresión', 'clustering', 'asociación',
            'random forest', 'naive bayes', 'svm', 'decision tree', 'neural network',
            'big data', 'business intelligence', 'analytics', 'estadística',
            'base de datos', 'sql', 'nosql', 'visualización', 'dashboard'
        ]
        
        found_concepts = []
        for concept in data_mining_concepts:
            if concept in text:
                found_concepts.append(concept)
                
        return found_concepts
    
    def search_content(self, query, top_k=5):
        """Busca contenido relevante basado en una consulta"""
        if not self.content_index:
            return []
        
        query_clean = self.clean_text(query)
        results = []
        
        for content_id, content_data in self.content_index.items():
            # Calcular similitud simple basada en palabras comunes
            content_words = set(content_data['text'].split())
            query_words = set(query_clean.split())
            
            intersection = content_words.intersection(query_words)
            union = content_words.union(query_words)
            
            if union:
                similarity = len(intersection) / len(union)
                results.append({
                    'content_id': content_id,
                    'similarity': similarity,
                    'content': content_data,
                    'matched_concepts': [c for c in content_data['concepts'] if c in query_clean]
                })
        
        # Ordenar por similitud
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

class QuestionAnsweringSystem:
    """Sistema avanzado de preguntas y respuestas"""
    
    def __init__(self, content_processor):
        self.content_processor = content_processor
        self.qa_history = []
        
        # Base de conocimiento expandida
        self.knowledge_base = {
            # Conceptos básicos
            "minería de datos": {
                "definition": "Proceso de descubrir patrones en grandes conjuntos de datos mediante métodos de inteligencia artificial, machine learning y estadística.",
                "keywords": ["patterns", "data", "discovery", "algorithms"],
                "related": ["machine learning", "data warehouse", "analytics"]
            },
            "data warehouse": {
                "definition": "Sistema de almacenamiento centralizado que integra datos de múltiples fuentes para análisis y toma de decisiones.",
                "keywords": ["storage", "integration", "analysis", "olap"],
                "related": ["etl", "olap", "business intelligence"]
            },
            "machine learning": {
                "definition": "Rama de la IA que permite a las máquinas aprender patrones de los datos sin programación explícita.",
                "keywords": ["learning", "algorithms", "prediction", "models"],
                "related": ["supervised learning", "unsupervised learning", "neural networks"]
            },
            "olap": {
                "definition": "Tecnología que permite análisis multidimensional rápido de datos empresariales.",
                "keywords": ["multidimensional", "analysis", "cubes", "aggregation"],
                "related": ["molap", "rolap", "holap", "data warehouse"]
            },
            "etl": {
                "definition": "Proceso de Extracción, Transformación y Carga de datos de sistemas fuente a destino.",
                "keywords": ["extraction", "transformation", "loading", "integration"],
                "related": ["data warehouse", "data integration", "data pipeline"]
            },
            "clasificación": {
                "definition": "Técnica de machine learning supervisado que predice categorías o clases de nuevos datos.",
                "keywords": ["supervised", "categories", "prediction", "labels"],
                "related": ["random forest", "svm", "naive bayes", "decision tree"]
            },
            "clustering": {
                "definition": "Técnica de aprendizaje no supervisado que agrupa datos similares.",
                "keywords": ["unsupervised", "groups", "similarity", "patterns"],
                "related": ["k-means", "hierarchical", "dbscan"]
            }
        }
    
    def process_question(self, question):
        """Procesa una pregunta y genera respuesta inteligente"""
        question_clean = question.lower().strip()
        
        # Buscar coincidencias directas
        best_match = self.find_best_match(question_clean)
        
        if best_match:
            response = self.generate_response(best_match, question_clean)
        else:
            # Buscar en contenido indexado
            search_results = self.content_processor.search_content(question_clean)
            if search_results:
                response = self.generate_contextual_response(search_results, question_clean)
            else:
                response = self.generate_fallback_response(question_clean)
        
        # Guardar en historial
        self.qa_history.append({
            'question': question,
            'response': response,
            'timestamp': datetime.now()
        })
        
        return response
    
    def find_best_match(self, question):
        """Encuentra la mejor coincidencia en la base de conocimiento"""
        best_score = 0
        best_match = None
        
        for concept, data in self.knowledge_base.items():
            score = 0
            
            # Puntuación por coincidencia directa del concepto
            if concept in question:
                score += 10
            
            # Puntuación por palabras clave
            for keyword in data['keywords']:
                if keyword in question:
                    score += 2
            
            # Puntuación por conceptos relacionados
            for related in data['related']:
                if related in question:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = concept
        
        return best_match if best_score > 0 else None
    
    def generate_response(self, concept, question):
        """Genera respuesta basada en el concepto encontrado"""
        data = self.knowledge_base[concept]
        
        response = f"**{concept.title()}**\n\n"
        response += f"{data['definition']}\n\n"
        
        if data['related']:
            response += f"**Conceptos relacionados:** {', '.join(data['related'])}"
        
        return response
    
    def generate_contextual_response(self, search_results, question):
        """Genera respuesta basada en contenido indexado"""
        response = "Basado en el contenido analizado:\n\n"
        
        for result in search_results[:2]:  # Top 2 resultados
            concepts = result['matched_concepts']
            if concepts:
                response += f"**Conceptos relevantes encontrados:** {', '.join(concepts)}\n"
                response += f"**Similitud:** {result['similarity']:.2%}\n\n"
        
        response += "Te recomiendo revisar el material indexado para más detalles."
        return response
    
    def generate_fallback_response(self, question):
        """Genera respuesta cuando no se encuentra información específica"""
        return """No encontré información específica sobre tu consulta. 
        
**Sugerencias:**
- Verifica la ortografía de términos técnicos
- Intenta usar términos más generales como 'minería de datos' o 'machine learning'
- Consulta la sección de preguntas frecuentes
        
**Temas disponibles:** minería de datos, data warehouse, OLAP, machine learning, clasificación, clustering."""

class ModelEvaluator:
    """Evaluador avanzado de modelos de machine learning"""
    
    def __init__(self):
        self.evaluation_history = []
    
    def comprehensive_evaluation(self, models_results, X_test, y_test):
        """Evaluación comprehensiva de múltiples modelos"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report
        )
        
        evaluation_results = {}
        
        for model_name, result in models_results.items():
            y_pred = result['predictions']
            y_true = result['true_values']
            
            # Métricas básicas
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            
            # ROC AUC para clasificación binaria
            if len(np.unique(y_true)) == 2:
                try:
                    if hasattr(result['model'], 'predict_proba'):
                        y_proba = result['model'].predict_proba(X_test)[:, 1]
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                    else:
                        metrics['roc_auc'] = None
                except:
                    metrics['roc_auc'] = None
            
            # Matriz de confusión
            cm = confusion_matrix(y_true, y_pred)
            
            # Reporte detallado
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            evaluation_results[model_name] = {
                'metrics': metrics,
                'confusion_matrix': cm,
                'classification_report': report,
                'model_object': result['model']
            }
        
        # Guardar en historial
        self.evaluation_history.append({
            'timestamp': datetime.now(),
            'results': evaluation_results,
            'num_models': len(models_results)
        })
        
        return evaluation_results
    
    def generate_recommendations(self, evaluation_results):
        """Genera recomendaciones basadas en la evaluación"""
        recommendations = []
        
        # Encontrar mejor modelo por métrica
        best_accuracy = max(evaluation_results.items(), 
                          key=lambda x: x[1]['metrics']['accuracy'])
        
        best_f1 = max(evaluation_results.items(), 
                     key=lambda x: x[1]['metrics']['f1_score'])
        
        recommendations.append(f"🏆 **Mejor precisión:** {best_accuracy[0]} ({best_accuracy[1]['metrics']['accuracy']:.4f})")
        recommendations.append(f"🎯 **Mejor F1-Score:** {best_f1[0]} ({best_f1[1]['metrics']['f1_score']:.4f})")
        
        # Análisis de rendimiento
        avg_accuracy = np.mean([r['metrics']['accuracy'] for r in evaluation_results.values()])
        
        if avg_accuracy > 0.9:
            recommendations.append("✅ **Excelente rendimiento general** - Los modelos muestran alta precisión")
        elif avg_accuracy > 0.8:
            recommendations.append("🟡 **Buen rendimiento** - Considera optimización de hiperparámetros")
        else:
            recommendations.append("🔴 **Rendimiento mejorable** - Revisa calidad de datos y feature engineering")
        
        # Recomendaciones específicas por tipo de modelo
        for model_name, results in evaluation_results.items():
            if model_name == "Random Forest" and results['metrics']['accuracy'] > 0.85:
                recommendations.append("🌲 Random Forest muestra buen rendimiento - Considera para producción")
            elif model_name == "SVM" and results['metrics']['accuracy'] < 0.7:
                recommendations.append("⚠️ SVM con bajo rendimiento - Verifica escalado de datos")
        
        return recommendations

class DataVisualizer:
    """Visualizador avanzado de datos y resultados"""
    
    def __init__(self):
        pass
    
    def create_model_comparison_dashboard(self, evaluation_results):
        """Crea dashboard completo de comparación de modelos"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Preparar datos
        models = list(evaluation_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, metric in enumerate(metrics):
            values = [evaluation_results[model]['metrics'][metric] for model in models]
            
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric.title(),
                    marker_color=colors[i],
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title_text="Comparación Detallada de Modelos",
            height=600,
            title_x=0.5
        )
        
        return fig
    
    def create_confusion_matrix_heatmap(self, confusion_matrix, model_name, class_names=None):
        """Crea heatmap de matriz de confusión"""
        import plotly.graph_objects as go
        
        if class_names is None:
            class_names = [f"Clase {i}" for i in range(len(confusion_matrix))]
        
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=confusion_matrix,
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f'Matriz de Confusión - {model_name}',
            xaxis_title='Predicciones',
            yaxis_title='Valores Reales',
            width=500,
            height=500
        )
        
        return fig
    
    def create_performance_radar_chart(self, evaluation_results):
        """Crea gráfico radar de rendimiento"""
        import plotly.graph_objects as go
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig = go.Figure()
        
        for model_name, results in evaluation_results.items():
            values = [results['metrics'][metric] for metric in metrics]
            values.append(values[0])  # Cerrar el polígono
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model_name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Comparación Multimétrica de Modelos"
        )
        
        return fig

# Funciones auxiliares para la interfaz
def create_download_link(data, filename, text):
    """Crea enlace de descarga para datos"""
    if isinstance(data, pd.DataFrame):
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
    else:
        b64 = base64.b64encode(str(data).encode()).decode()
    
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def format_metrics_table(evaluation_results):
    """Formatea tabla de métricas para visualización"""
    data = []
    for model_name, results in evaluation_results.items():
        metrics = results['metrics']
        data.append({
            'Modelo': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'ROC-AUC': f"{metrics.get('roc_auc', 'N/A')}" if metrics.get('roc_auc') else 'N/A'
        })
    
    return pd.DataFrame(data)

def export_model(model, model_name):
    """Exporta modelo entrenado"""
    import pickle
    
    model_data = {
        'model': model,
        'model_name': model_name,
        'timestamp': datetime.now(),
        'version': '1.0'
    }
    
    return pickle.dumps(model_data)