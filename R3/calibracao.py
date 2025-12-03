import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- Dados Fornecidos ---
# Padrão (Visor Local) - Eixo X
x_padrao = np.array([485.0, 449.0, 398.0, 350.0, 298.0, 252.0, 201.0, 147.0, 104.0, 65.0])
# Medidor (LIT-101) - Eixo Y
y_medido = np.array([480.5, 448.7, 395.3, 347.7, 297.9, 252.2, 202.0, 149.8, 105.8, 66.8])

# --- 1. Regressão Linear (Método dos Mínimos Quadrados) ---
# Modelo: y = a*x + b
slope, intercept, r_value, p_value, std_err = stats.linregress(x_padrao, y_medido)
a = slope
b = intercept

r_squared = r_value**2 # type: ignore

# Valores preditos pela reta
y_pred = a * x_padrao + b
# Resíduos (y_medido - y_pred)
residuos = y_medido - y_pred

# --- 2. Análise de Incerteza ---
n = len(x_padrao)
graus_liberdade = n - 2

# Desvio padrão dos resíduos (Sy)
sy = np.sqrt(np.sum(residuos**2) / graus_liberdade)

# Incerteza Padrão (u)
# Conforme instrução da Aula 11: desconsiderando incerteza dos parâmetros e do padrão.
# A incerteza padrão da indicação corrigida é dada por Sy / a
u_padrao = sy / a

# Fator de Abrangência (t-student)
# Para 95.45% de confiança e graus de liberdade = n-2
# Consultando a tabela ou calculando (tinv)
confianca = 0.9545
alpha = 1 - confianca
t_student = stats.t.ppf(1 - alpha/2, graus_liberdade)

# Incerteza Expandida (U)
U_expandida = t_student * u_padrao

# --- 3. Linearidade e Erro ---
# Span do instrumento (considerado 0 a 500 mm conforme especificações do tanque)
span = 500.0

# Linearidade (Máximo resíduo absoluto)
linearidade_abs = np.max(np.abs(residuos))
linearidade_perc = (linearidade_abs / span) * 100

# Erro Fiducial (Máximo erro absoluto em relação ao padrão / Span)
erro_absoluto_max = np.max(np.abs(y_medido - x_padrao))
erro_fiducial = (erro_absoluto_max / span) * 100

# Limites de incerteza para plotagem
limite_superior = y_pred + U_expandida
limite_inferior = y_pred - U_expandida

# --- 4. Plotagem do Gráfico ---
plt.figure(figsize=(10, 6))
plt.scatter(x_padrao, y_medido, color='blue', label='Dados Medidos')
plt.plot(x_padrao, y_pred, color='red', linestyle='--', label=f'Reta Ajustada: y = {a:.4f}x + {b:.4f}')
plt.fill_between(x_padrao, limite_inferior, limite_superior, color='gray', alpha=0.3, label='Incerteza Expandida (U)')
plt.title('Curva de Calibração: LIT-101 vs Padrão')
plt.xlabel('Nível Padrão (mm)')
plt.ylabel('Indicação LIT-101 (mm)')
plt.legend()
plt.grid(True)
plt.text(100, 400, f'$R^2 = {r_squared:.5f}$', fontsize=12)

# Salvar gráfico
plt.savefig('calibracao_lit101.png')
print("Gráfico salvo como 'calibracao_lit101.png'")

# --- 5. Exibição dos Resultados para o Relatório ---
print("-" * 30)
print("RESULTADOS PARA O RELATÓRIO")
print("-" * 30)
print(f"Equação da Reta: y = {a:.4f} * x + {b:.4f}")
print(f"Coeficiente de Determinação (R²): {r_squared:.5f}")
print(f"Desvio Padrão dos Resíduos (Sy): {sy:.4f} mm")
print(f"Graus de Liberdade (v): {graus_liberdade}")
print(f"Fator t-Student (95.45%): {t_student:.4f}")
print(f"Incerteza Padrão (u): {u_padrao:.4f} mm")
print(f"Incerteza Expandida (U): {U_expandida:.4f} mm")
print(f"Linearidade: {linearidade_abs:.4f} mm ({linearidade_perc:.2f}%)")
print(f"Erro Máximo Fiducial: {erro_fiducial:.2f}% (Erro Max: {erro_absoluto_max:.2f} mm)")
print("-" * 30)