# Plano de Deploy em Produção - Dashboard e Modelo de Previsão do Petróleo Brent

Este documento detalha o plano para deploy em produção do dashboard interativo e modelo de previsão do preço do petróleo Brent, garantindo disponibilidade, escalabilidade, segurança e manutenção contínua.

## 1. Arquitetura da Solução

### 1.1 Componentes Principais

- **Dashboard Streamlit**: Interface interativa para visualização de dados e previsões
- **Modelo LSTM**: Modelo de Machine Learning para previsão do preço do petróleo
- **Banco de Dados**: Armazenamento dos dados históricos e previsões
- **Pipeline de Atualização**: Mecanismo para atualização automática dos dados e retreinamento do modelo

### 1.2 Diagrama de Arquitetura

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Fonte de Dados │────▶│  Pipeline ETL   │────▶│  Banco de Dados │
│    (IPEA API)   │     │                 │     │                 │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│    Usuários     │◀───▶│    Dashboard    │◀───▶│  Modelo LSTM    │
│                 │     │    Streamlit    │     │                 │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 2. Infraestrutura

### 2.1 Opções de Hospedagem

#### Opção 1: Streamlit Cloud (Recomendada para MVP)
- **Vantagens**: Deploy rápido, gerenciamento simplificado, integração nativa com GitHub
- **Desvantagens**: Limitações de personalização, recursos computacionais limitados
- **Custo estimado**: Gratuito para uso básico, $10-50/mês para recursos adicionais

#### Opção 2: AWS/GCP/Azure
- **Vantagens**: Alta escalabilidade, flexibilidade, recursos computacionais robustos
- **Desvantagens**: Configuração mais complexa, custo mais elevado
- **Custo estimado**: $50-200/mês dependendo da configuração

#### Opção 3: VPS Dedicado
- **Vantagens**: Controle total, custo previsível
- **Desvantagens**: Gerenciamento manual, escalabilidade limitada
- **Custo estimado**: $20-100/mês

### 2.2 Requisitos de Infraestrutura

- **CPU**: 2+ núcleos para processamento do modelo
- **RAM**: Mínimo 4GB, recomendado 8GB
- **Armazenamento**: 10GB para aplicação, dados e modelo
- **Rede**: Largura de banda suficiente para múltiplos usuários simultâneos

## 3. Containerização com Docker

### 3.1 Estrutura do Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar arquivos da aplicação
COPY . .

# Expor porta para o Streamlit
EXPOSE 8501

# Comando para iniciar a aplicação
CMD ["streamlit", "run", "dashboard_com_modelo.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 3.2 Docker Compose para Ambiente Completo

```yaml
version: '3'

services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./dados_processados:/app/dados_processados
      - ./modelo:/app/modelo
    restart: always
    environment:
      - TZ=America/Sao_Paulo
```

## 4. Pipeline de Dados e Atualização do Modelo

### 4.1 Atualização de Dados

- **Frequência**: Diária (após fechamento do mercado)
- **Fonte**: API do IPEA ou web scraping automatizado
- **Processo**:
  1. Coleta de novos dados
  2. Validação e limpeza
  3. Integração ao banco de dados histórico
  4. Geração de novas features

### 4.2 Retreinamento do Modelo

- **Frequência**: Semanal ou mensal
- **Processo**:
  1. Extração dos dados atualizados
  2. Retreinamento do modelo LSTM
  3. Avaliação de performance
  4. Substituição do modelo em produção se performance melhorar
  5. Geração de novas previsões

### 4.3 Script de Automação

```python
# pipeline_atualizacao.py
import os
import pandas as pd
from datetime import datetime
import subprocess
import logging

# Configurar logging
logging.basicConfig(
    filename='pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def atualizar_dados():
    """Atualiza os dados históricos do petróleo Brent."""
    try:
        logging.info("Iniciando atualização de dados")
        # Executar script de coleta de dados
        subprocess.run(["python", "scraper_petroleo.py"], check=True)
        logging.info("Dados atualizados com sucesso")
        return True
    except Exception as e:
        logging.error(f"Erro na atualização de dados: {e}")
        return False

def retreinar_modelo():
    """Retreina o modelo LSTM com os dados atualizados."""
    try:
        logging.info("Iniciando retreinamento do modelo")
        # Executar script de treinamento do modelo
        subprocess.run(["python", "modelo_previsao.py"], check=True)
        logging.info("Modelo retreinado com sucesso")
        return True
    except Exception as e:
        logging.error(f"Erro no retreinamento do modelo: {e}")
        return False

def main():
    # Verificar dia da semana (1-7, onde 1 é segunda-feira)
    dia_semana = datetime.now().isoweekday()
    
    # Atualizar dados diariamente
    sucesso_dados = atualizar_dados()
    
    # Retreinar modelo semanalmente (aos domingos)
    if dia_semana == 7 and sucesso_dados:
        retreinar_modelo()

if __name__ == "__main__":
    main()
```

## 5. Segurança e Monitoramento

### 5.1 Medidas de Segurança

- **HTTPS**: Configurar certificado SSL para comunicação segura
- **Autenticação**: Implementar sistema de login para acesso ao dashboard (opcional)
- **Backup**: Backup diário dos dados e modelos
- **Logs**: Registro detalhado de atividades e erros

### 5.2 Monitoramento

- **Performance da Aplicação**: Tempo de resposta, uso de recursos
- **Performance do Modelo**: Métricas de erro (RMSE, MAE, MAPE)
- **Alertas**: Notificações para falhas na atualização de dados ou degradação da performance do modelo

### 5.3 Ferramentas de Monitoramento

- **Prometheus/Grafana**: Para métricas de infraestrutura
- **MLflow**: Para rastreamento de experimentos e versões do modelo
- **Sentry**: Para monitoramento de erros em tempo real

## 6. Processo de Deploy

### 6.1 Deploy Inicial (MVP)

1. **Preparação**:
   - Finalizar desenvolvimento e testes locais
   - Gerar arquivo `requirements.txt` com todas as dependências
   - Criar repositório Git para versionamento

2. **Deploy no Streamlit Cloud**:
   - Conectar repositório GitHub ao Streamlit Cloud
   - Configurar variáveis de ambiente necessárias
   - Realizar deploy inicial

3. **Validação**:
   - Testar todas as funcionalidades no ambiente de produção
   - Verificar performance e tempo de resposta
   - Corrigir eventuais problemas

### 6.2 Deploy em Ambiente Corporativo (Fase 2)

1. **Preparação da Infraestrutura**:
   - Provisionar servidor/VM conforme requisitos
   - Configurar rede, firewall e DNS

2. **Containerização**:
   - Construir imagem Docker
   - Testar container localmente

3. **Deploy**:
   - Transferir imagem para servidor
   - Iniciar container com Docker Compose
   - Configurar proxy reverso (Nginx/Traefik) e SSL

4. **Automação**:
   - Configurar pipeline CI/CD para atualizações automáticas
   - Implementar scripts de atualização de dados e modelo

## 7. Manutenção e Evolução

### 7.1 Manutenção Regular

- **Atualizações de Dependências**: Mensal
- **Backup de Dados**: Diário
- **Revisão de Performance**: Semanal
- **Validação de Previsões**: Comparação mensal entre previsões e valores reais

### 7.2 Evolução Planejada

- **Fase 1 (MVP)**: Dashboard básico com modelo LSTM
- **Fase 2**: Incorporação de variáveis exógenas ao modelo
- **Fase 3**: Implementação de modelos ensemble para melhorar precisão
- **Fase 4**: Adição de análises de cenários e simulações

## 8. Requisitos para Produção

### 8.1 Arquivo requirements.txt

```
streamlit==1.32.0
pandas==2.1.3
numpy==2.1.3
matplotlib==3.8.2
seaborn==0.13.1
plotly==5.18.0
tensorflow==2.19.0
scikit-learn==1.6.1
joblib==1.5.0
requests==2.31.0
beautifulsoup4==4.12.2
markdown==3.8
pillow==10.2.0
```

### 8.2 Estrutura de Diretórios

```
projeto_petroleo/
├── dashboard_com_modelo.py     # Aplicação Streamlit principal
├── modelo_previsao.py          # Script de treinamento do modelo
├── scraper_petroleo.py         # Script de coleta de dados
├── pipeline_atualizacao.py     # Script de automação
├── requirements.txt            # Dependências
├── Dockerfile                  # Configuração do container
├── docker-compose.yml          # Configuração do ambiente
├── README.md                   # Documentação
├── dados_processados/          # Dados processados
│   ├── petroleo_brent_processado.csv
│   ├── eventos_importantes.csv
│   └── previsao_futura.csv
├── modelo/                     # Artefatos do modelo
│   ├── modelo_lstm.h5
│   ├── scaler_X.pkl
│   ├── scaler_y.pkl
│   ├── parametros.pkl
│   └── metricas_por_horizonte.csv
├── visualizacoes/              # Imagens e gráficos
└── documentacao/               # Documentação detalhada
    └── documentacao_modelo.md
```

## 9. Estimativa de Custos

### 9.1 Custos Iniciais (MVP)

- **Streamlit Cloud**: $0-50/mês
- **Desenvolvimento e Setup**: 40 horas de trabalho

### 9.2 Custos Mensais (Ambiente Corporativo)

- **Infraestrutura Cloud**: $50-200/mês
- **Manutenção e Suporte**: 10 horas/mês
- **Atualizações e Melhorias**: 20 horas/mês

## 10. Cronograma de Implementação

### 10.1 MVP (2-4 semanas)

- **Semana 1**: Finalização do desenvolvimento e testes locais
- **Semana 2**: Deploy no Streamlit Cloud e validação
- **Semana 3**: Ajustes finais e documentação
- **Semana 4**: Treinamento dos usuários e entrega

## 11. Conclusão

Este plano de deploy fornece um roteiro completo para colocar em produção o dashboard interativo e o modelo de previsão do preço do petróleo Brent. A abordagem em fases permite uma implementação gradual, começando com um MVP no Streamlit Cloud e evoluindo para uma solução corporativa robusta conforme necessário.

A solução proposta equilibra facilidade de implementação, custo e escalabilidade, permitindo que o cliente obtenha valor rapidamente enquanto mantém a flexibilidade para expansões futuras.

        # Título acima do link
        st.subheader("Notebook utilizado inicialmente como teste de previsão")

        st.markdown(
        '<a href="(https://github.com/marloncabral/TechChallenge/blob/main/documentacao/plano_deploy.md)" target="_blank">🔗 Acesse o notebook completo no GitHub</a>',
        unsafe_allow_html=True

