# ============================================================================
# CALCULADORA DE IMC CON MACHINE LEARNING - INTERFAZ VISUAL
# ============================================================================
# Sistema que aprende a calcular IMC usando regresión lineal con interfaz gráfica
# ============================================================================

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class BMIMLCalculator:
    """Calculadora de IMC que utiliza Machine Learning."""
    
    def __init__(self):
        self.altura_data = []
        self.peso_data = []
        self.imc_data = []
        
        self.coef_altura = 0.0
        self.coef_peso = 0.0
        self.intercepto = 0.0
        
        self.is_trained = False
        self.num_datos_entrenamiento = 0
        
        self.mse = 0.0
        self.r2 = 0.0
        self.mae = 0.0
    
    def calcular_imc_real(self, altura, peso):
        if altura <= 0:
            return 0.0
        return peso / (altura * altura)
    
    def clasificar_imc(self, imc):
        if imc < 18.5:
            return "Peso insuficiente"
        elif imc < 25.0:
            return "Peso normal"
        elif imc < 30.0:
            return "Sobrepeso"
        else:
            return "Obesidad"
    
    def agregar_dato(self, altura, peso):
        """Agrega un dato de entrenamiento."""
        if altura <= 0 or altura > 3.0 or peso <= 0 or peso > 500:
            return False, "Valores fuera de rango válido."
        
        imc_real = self.calcular_imc_real(altura, peso)
        
        self.altura_data.append(altura)
        self.peso_data.append(peso)
        self.imc_data.append(imc_real)
        self.num_datos_entrenamiento += 1
        
        return True, f"Dato agregado. IMC: {imc_real:.2f}"
    
    def entrenar_modelo(self):
        """Entrena el modelo de regresión múltiple."""
        if len(self.altura_data) < 3:
            return False, "Se necesitan al menos 3 datos para entrenar el modelo."
        
        n = len(self.altura_data)
        
        sum_altura = sum(self.altura_data)
        sum_peso = sum(self.peso_data)
        sum_imc = sum(self.imc_data)
        
        sum_altura2 = sum(h * h for h in self.altura_data)
        sum_peso2 = sum(p * p for p in self.peso_data)
        sum_altura_peso = sum(h * p for h, p in zip(self.altura_data, self.peso_data))
        
        sum_altura_imc = sum(h * i for h, i in zip(self.altura_data, self.imc_data))
        sum_peso_imc = sum(p * i for p, i in zip(self.peso_data, self.imc_data))
        
        try:
            # Correlación altura-IMC
            corr_altura_imc = (n * sum_altura_imc - sum_altura * sum_imc) / \
                             math.sqrt((n * sum_altura2 - sum_altura * sum_altura) * 
                                     (n * sum(i*i for i in self.imc_data) - sum_imc * sum_imc))
            
            # Correlación peso-IMC
            corr_peso_imc = (n * sum_peso_imc - sum_peso * sum_imc) / \
                           math.sqrt((n * sum_peso2 - sum_peso * sum_peso) * 
                                   (n * sum(i*i for i in self.imc_data) - sum_imc * sum_imc))
            
            self.coef_altura = corr_altura_imc * 10
            self.coef_peso = corr_peso_imc * 0.5
            self.intercepto = (sum_imc - self.coef_altura * sum_altura - self.coef_peso * sum_peso) / n
            
            self.is_trained = True
            
            # Calcular métricas
            self._calcular_metricas()
            
            ecuacion = f"IMC = {self.coef_altura:.4f} * altura + {self.coef_peso:.4f} * peso + {self.intercepto:.4f}"
            return True, f"Modelo entrenado exitosamente!\n{ecuacion}"
            
        except Exception as e:
            return False, f"Error durante el entrenamiento: {e}"
    
    def _calcular_metricas(self):
        """Calcula las métricas de evaluación."""
        predicciones = []
        errores_absolutos = []
        errores_cuadraticos = []
        
        for i in range(len(self.altura_data)):
            altura = self.altura_data[i]
            peso = self.peso_data[i]
            imc_real = self.imc_data[i]
            
            imc_predicho = self.coef_altura * altura + self.coef_peso * peso + self.intercepto
            predicciones.append(imc_predicho)
            
            error_absoluto = abs(imc_real - imc_predicho)
            error_cuadratico = (imc_real - imc_predicho) ** 2
            
            errores_absolutos.append(error_absoluto)
            errores_cuadraticos.append(error_cuadratico)
        
        self.mae = sum(errores_absolutos) / len(errores_absolutos)
        self.mse = sum(errores_cuadraticos) / len(errores_cuadraticos)
        
        imc_promedio = sum(self.imc_data) / len(self.imc_data)
        ss_tot = sum((imc - imc_promedio) ** 2 for imc in self.imc_data)
        ss_res = sum(errores_cuadraticos)
        
        if ss_tot > 0:
            self.r2 = 1 - (ss_res / ss_tot)
        else:
            self.r2 = 1.0
    
    def predecir_imc(self, altura, peso):
        """Predice el IMC usando el modelo entrenado."""
        if not self.is_trained:
            return None, "El modelo debe ser entrenado primero."
        
        if altura <= 0 or altura > 3.0 or peso <= 0 or peso > 500:
            return None, "Valores fuera de rango válido."
        
        imc_ml = self.coef_altura * altura + self.coef_peso * peso + self.intercepto
        imc_real = self.calcular_imc_real(altura, peso)
        
        return {
            'imc_ml': imc_ml,
            'imc_real': imc_real,
            'diferencia': abs(imc_ml - imc_real),
            'clasificacion_ml': self.clasificar_imc(imc_ml),
            'clasificacion_real': self.clasificar_imc(imc_real)
        }, "Predicción exitosa"

class BMIMLApp:
    """Interfaz gráfica para la calculadora de IMC con ML."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Calculadora IMC con Machine Learning")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Inicializar calculadora
        self.calculator = BMIMLCalculator()
        
        # Crear interfaz
        self.crear_interfaz()
        
        # Variables de color para clasificaciones
        self.colores_imc = {
            "Peso insuficiente": "#3498db",
            "Peso normal": "#27ae60",
            "Sobrepeso": "#f39c12",
            "Obesidad": "#e74c3c"
        }
    
    def crear_interfaz(self):
        """Crea todos los elementos de la interfaz."""
        # Título principal
        titulo = tk.Label(self.root, text="🤖 Calculadora IMC con Machine Learning", 
                         font=("Arial", 18, "bold"), bg='#f0f0f0', fg='#2c3e50')
        titulo.pack(pady=10)
        
        # Crear notebook para pestañas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Pestañas
        self.crear_pestaña_entrenamiento()
        self.crear_pestaña_prediccion()
        self.crear_pestaña_datos()
        self.crear_pestaña_metricas()
    
    def crear_pestaña_entrenamiento(self):
        """Crea la pestaña para agregar datos y entrenar."""
        frame_entrenamiento = ttk.Frame(self.notebook)
        self.notebook.add(frame_entrenamiento, text="📊 Entrenamiento")
        
        # Frame para agregar datos
        frame_datos = tk.LabelFrame(frame_entrenamiento, text="Agregar Datos de Entrenamiento", 
                                   font=("Arial", 12, "bold"), padx=10, pady=10)
        frame_datos.pack(fill='x', padx=10, pady=5)
        
        # Campos de entrada
        tk.Label(frame_datos, text="Altura (m):", font=("Arial", 10)).grid(row=0, column=0, sticky='w', pady=5)
        self.entry_altura_train = tk.Entry(frame_datos, font=("Arial", 11), width=15)
        self.entry_altura_train.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(frame_datos, text="Peso (kg):", font=("Arial", 10)).grid(row=0, column=2, sticky='w', pady=5)
        self.entry_peso_train = tk.Entry(frame_datos, font=("Arial", 11), width=15)
        self.entry_peso_train.grid(row=0, column=3, padx=5, pady=5)
        
        # Botón agregar dato
        btn_agregar = tk.Button(frame_datos, text="➕ Agregar Dato", 
                               command=self.agregar_dato, font=("Arial", 10, "bold"),
                               bg='#3498db', fg='white', relief='flat', padx=20)
        btn_agregar.grid(row=0, column=4, padx=10, pady=5)
        
        # Información del último dato
        self.label_ultimo_dato = tk.Label(frame_datos, text="", font=("Arial", 10), fg='#27ae60')
        self.label_ultimo_dato.grid(row=1, column=0, columnspan=5, pady=5)
        
        # Frame para entrenamiento
        frame_train = tk.LabelFrame(frame_entrenamiento, text="Entrenar Modelo", 
                                   font=("Arial", 12, "bold"), padx=10, pady=10)
        frame_train.pack(fill='x', padx=10, pady=5)
        
        # Información del modelo
        self.label_info_modelo = tk.Label(frame_train, text="Datos disponibles: 0\nModelo: No entrenado", 
                                         font=("Arial", 10), justify='left')
        self.label_info_modelo.pack(anchor='w', pady=5)
        
        # Botón entrenar
        btn_entrenar = tk.Button(frame_train, text="🎯 Entrenar Modelo", 
                                command=self.entrenar_modelo, font=("Arial", 12, "bold"),
                                bg='#e74c3c', fg='white', relief='flat', padx=20, pady=5)
        btn_entrenar.pack(pady=10)
        
        # Ecuación del modelo
        self.label_ecuacion = tk.Label(frame_train, text="", font=("Arial", 10), 
                                      fg='#8e44ad', wraplength=600)
        self.label_ecuacion.pack(pady=5)
    
    def crear_pestaña_prediccion(self):
        """Crea la pestaña para hacer predicciones."""
        frame_prediccion = ttk.Frame(self.notebook)
        self.notebook.add(frame_prediccion, text="🔮 Predicción")
        
        # Frame de entrada
        frame_entrada = tk.LabelFrame(frame_prediccion, text="Datos para Predicción", 
                                     font=("Arial", 12, "bold"), padx=10, pady=10)
        frame_entrada.pack(fill='x', padx=10, pady=5)
        
        # Campos
        tk.Label(frame_entrada, text="Altura (m):", font=("Arial", 10)).grid(row=0, column=0, sticky='w', pady=5)
        self.entry_altura_pred = tk.Entry(frame_entrada, font=("Arial", 11), width=15)
        self.entry_altura_pred.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(frame_entrada, text="Peso (kg):", font=("Arial", 10)).grid(row=0, column=2, sticky='w', pady=5)
        self.entry_peso_pred = tk.Entry(frame_entrada, font=("Arial", 11), width=15)
        self.entry_peso_pred.grid(row=0, column=3, padx=5, pady=5)
        
        # Botón predecir
        btn_predecir = tk.Button(frame_entrada, text="🔍 Predecir IMC", 
                                command=self.predecir_imc, font=("Arial", 10, "bold"),
                                bg='#27ae60', fg='white', relief='flat', padx=20)
        btn_predecir.grid(row=0, column=4, padx=10, pady=5)
        
        # Frame de resultados
        frame_resultados = tk.LabelFrame(frame_prediccion, text="Resultados", 
                                        font=("Arial", 12, "bold"), padx=10, pady=10)
        frame_resultados.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Labels para resultados
        self.label_resultados = tk.Text(frame_resultados, height=15, font=("Arial", 11), 
                                       wrap='word', state='disabled')
        self.label_resultados.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Scrollbar para resultados
        scrollbar_res = tk.Scrollbar(frame_resultados, command=self.label_resultados.yview)
        self.label_resultados.config(yscrollcommand=scrollbar_res.set)
    
    def crear_pestaña_datos(self):
        """Crea la pestaña para ver los datos de entrenamiento."""
        frame_datos = ttk.Frame(self.notebook)
        self.notebook.add(frame_datos, text="📋 Datos")
        
        # Frame principal
        frame_tabla = tk.LabelFrame(frame_datos, text="Datos de Entrenamiento", 
                                   font=("Arial", 12, "bold"), padx=10, pady=10)
        frame_tabla.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Treeview para mostrar datos
        columnas = ("ID", "Altura (m)", "Peso (kg)", "IMC", "Clasificación")
        self.tree_datos = ttk.Treeview(frame_tabla, columns=columnas, show='headings', height=15)
        
        # Configurar columnas
        for col in columnas:
            self.tree_datos.heading(col, text=col)
            self.tree_datos.column(col, width=120, anchor='center')
        
        # Scrollbars
        scrollbar_v = ttk.Scrollbar(frame_tabla, orient='vertical', command=self.tree_datos.yview)
        scrollbar_h = ttk.Scrollbar(frame_tabla, orient='horizontal', command=self.tree_datos.xview)
        self.tree_datos.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        # Posicionar elementos
        self.tree_datos.pack(side='left', fill='both', expand=True)
        scrollbar_v.pack(side='right', fill='y')
        scrollbar_h.pack(side='bottom', fill='x')
        
        # Botón actualizar
        btn_actualizar_datos = tk.Button(frame_datos, text="🔄 Actualizar Datos", 
                                        command=self.actualizar_tabla_datos, 
                                        font=("Arial", 10, "bold"), bg='#3498db', fg='white')
        btn_actualizar_datos.pack(pady=5)
    
    def crear_pestaña_metricas(self):
        """Crea la pestaña para mostrar métricas y gráficos."""
        frame_metricas = ttk.Frame(self.notebook)
        self.notebook.add(frame_metricas, text="📈 Métricas")
        
        # Frame de métricas
        frame_info = tk.LabelFrame(frame_metricas, text="Métricas del Modelo", 
                                  font=("Arial", 12, "bold"), padx=10, pady=10)
        frame_info.pack(fill='x', padx=10, pady=5)
        
        self.label_metricas = tk.Label(frame_info, text="Entrena el modelo primero para ver métricas", 
                                      font=("Arial", 11), justify='left')
        self.label_metricas.pack(anchor='w', pady=5)
        
        # Frame para gráfico
        frame_grafico = tk.LabelFrame(frame_metricas, text="Visualización", 
                                     font=("Arial", 12, "bold"), padx=10, pady=10)
        frame_grafico.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Botón para mostrar gráfico
        btn_grafico = tk.Button(frame_grafico, text="📊 Mostrar Gráfico de Predicciones", 
                               command=self.mostrar_grafico, font=("Arial", 10, "bold"),
                               bg='#9b59b6', fg='white', relief='flat', padx=20)
        btn_grafico.pack(pady=10)
        
        # Frame para el gráfico
        self.frame_plot = tk.Frame(frame_grafico)
        self.frame_plot.pack(fill='both', expand=True)
    
    def agregar_dato(self):
        """Agrega un dato de entrenamiento."""
        try:
            altura = float(self.entry_altura_train.get())
            peso = float(self.entry_peso_train.get())
            
            exito, mensaje = self.calculator.agregar_dato(altura, peso)
            
            if exito:
                self.label_ultimo_dato.config(text=f"✓ {mensaje}", fg='#27ae60')
                self.entry_altura_train.delete(0, 'end')
                self.entry_peso_train.delete(0, 'end')
                self.actualizar_info_modelo()
                self.actualizar_tabla_datos()
            else:
                messagebox.showerror("Error", mensaje)
                
        except ValueError:
            messagebox.showerror("Error", "Ingrese valores numéricos válidos")
    
    def entrenar_modelo(self):
        """Entrena el modelo de ML."""
        exito, mensaje = self.calculator.entrenar_modelo()
        
        if exito:
            self.label_ecuacion.config(text=mensaje)
            self.actualizar_info_modelo()
            self.actualizar_metricas()
            messagebox.showinfo("Éxito", "Modelo entrenado exitosamente!")
        else:
            messagebox.showerror("Error", mensaje)
    
    def predecir_imc(self):
        """Hace una predicción de IMC."""
        try:
            altura = float(self.entry_altura_pred.get())
            peso = float(self.entry_peso_pred.get())
            
            resultado, mensaje = self.calculator.predecir_imc(altura, peso)
            
            if resultado:
                # Mostrar resultados en el text widget
                self.label_resultados.config(state='normal')
                self.label_resultados.delete(1.0, 'end')
                
                texto_resultado = f"""
=== RESULTADOS DE PREDICCIÓN ===

Datos ingresados:
• Altura: {altura:.2f} m
• Peso: {peso:.1f} kg

Resultados:
• IMC predicho por ML: {resultado['imc_ml']:.2f}
• IMC fórmula real: {resultado['imc_real']:.2f}
• Diferencia: {resultado['diferencia']:.2f}

Clasificaciones:
• Según modelo ML: {resultado['clasificacion_ml']}
• Según fórmula real: {resultado['clasificacion_real']}

Interpretación:
"""
                
                if resultado['diferencia'] < 0.5:
                    texto_resultado += "✓ Excelente: El modelo ML predice con muy alta precisión"
                elif resultado['diferencia'] < 1.0:
                    texto_resultado += "✓ Bueno: El modelo ML tiene buena precisión"
                elif resultado['diferencia'] < 2.0:
                    texto_resultado += "⚠ Regular: El modelo ML tiene precisión aceptable"
                else:
                    texto_resultado += "❌ El modelo necesita más datos de entrenamiento"
                
                self.label_resultados.insert(1.0, texto_resultado)
                self.label_resultados.config(state='disabled')
                
            else:
                messagebox.showerror("Error", mensaje)
                
        except ValueError:
            messagebox.showerror("Error", "Ingrese valores numéricos válidos")
    
    def actualizar_info_modelo(self):
        """Actualiza la información del modelo."""
        estado = "Entrenado ✓" if self.calculator.is_trained else "No entrenado"
        info = f"Datos disponibles: {self.calculator.num_datos_entrenamiento}\nModelo: {estado}"
        self.label_info_modelo.config(text=info)
    
    def actualizar_tabla_datos(self):
        """Actualiza la tabla de datos de entrenamiento."""
        # Limpiar tabla
        for item in self.tree_datos.get_children():
            self.tree_datos.delete(item)
        
        # Agregar datos
        for i in range(len(self.calculator.altura_data)):
            altura = self.calculator.altura_data[i]
            peso = self.calculator.peso_data[i]
            imc = self.calculator.imc_data[i]
            clasificacion = self.calculator.clasificar_imc(imc)
            
            self.tree_datos.insert('', 'end', values=(
                i+1, f"{altura:.2f}", f"{peso:.1f}", f"{imc:.2f}", clasificacion
            ))
    
    def actualizar_metricas(self):
        """Actualiza las métricas del modelo."""
        if self.calculator.is_trained:
            texto_metricas = f"""Métricas de Evaluación:

• Error Absoluto Medio (MAE): {self.calculator.mae:.4f}
• Error Cuadrático Medio (MSE): {self.calculator.mse:.4f}
• Coeficiente de Determinación (R²): {self.calculator.r2:.4f}

Interpretación:"""
            
            if self.calculator.mae < 1.0:
                texto_metricas += "\n✓ Excelente: El modelo predice IMC con muy alta precisión"
            elif self.calculator.mae < 2.0:
                texto_metricas += "\n✓ Bueno: El modelo tiene buena precisión"
            elif self.calculator.mae < 3.0:
                texto_metricas += "\n⚠ Regular: El modelo tiene precisión aceptable"
            else:
                texto_metricas += "\n❌ Pobre: El modelo necesita más datos"
            
            if self.calculator.r2 > 0.9:
                texto_metricas += "\n✓ El modelo explica más del 90% de la variabilidad"
            elif self.calculator.r2 > 0.7:
                texto_metricas += "\n✓ El modelo explica más del 70% de la variabilidad"
            else:
                texto_metricas += "\n⚠ El modelo explica menos del 70% de la variabilidad"
            
            self.label_metricas.config(text=texto_metricas)
    
    def mostrar_grafico(self):
        """Muestra un gráfico de las predicciones vs valores reales."""
        if not self.calculator.is_trained:
            messagebox.showwarning("Advertencia", "Entrena el modelo primero")
            return
        
        if len(self.calculator.altura_data) == 0:
            messagebox.showwarning("Advertencia", "No hay datos para mostrar")
            return
        
        # Limpiar frame anterior
        for widget in self.frame_plot.winfo_children():
            widget.destroy()
        
        # Crear gráfico
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calcular predicciones
        predicciones = []
        for i in range(len(self.calculator.altura_data)):
            altura = self.calculator.altura_data[i]
            peso = self.calculator.peso_data[i]
            pred = (self.calculator.coef_altura * altura + 
                   self.calculator.coef_peso * peso + self.calculator.intercepto)
            predicciones.append(pred)
        
        # Gráfico 1: Predicciones vs Reales
        ax1.scatter(self.calculator.imc_data, predicciones, alpha=0.7, color='blue')
        ax1.plot([min(self.calculator.imc_data), max(self.calculator.imc_data)], 
                [min(self.calculator.imc_data), max(self.calculator.imc_data)], 
                'r--', label='Predicción perfecta')
        ax1.set_xlabel('IMC Real')
        ax1.set_ylabel('IMC Predicho')
        ax1.set_title('Predicciones vs Valores Reales')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Distribución de errores
        errores = [abs(real - pred) for real, pred in zip(self.calculator.imc_data, predicciones)]
        ax2.hist(errores, bins=min(10, len(errores)), alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Error Absoluto')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Errores')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Integrar gráfico en tkinter
        canvas = FigureCanvasTkAgg(fig, self.frame_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

def main():
    """Función principal."""
    root = tk.Tk()
    app = BMIMLApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
