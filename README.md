# Capa

- **Título do Projeto**: Sistema de Otimização de Rotores em Bombas Centrífugas com Machine Learning
- **Acadêmico**: Jairson Steinert
- **Professor Orientador**: Prof. Dr. Andrei Carniel
- **Curso**: Engenharia de Software
- **Data de Entrega**: 30 de novembro de 2025

---

# Resumo

Este documento apresenta a especificação técnica de um projeto de portfólio que visa desenvolver um sistema inteligente para otimização do ajuste de rotores/impulsores em bombas centrífugas utilizando técnicas de Machine Learning. O projeto aborda a limitação das fórmulas empíricas tradicionais de proporcionalidade, propondo a criação de um coeficiente de ajuste preditivo baseado em dados históricos de operação. A solução integra um pipeline completo de IA, desde a coleta e preparação de dados até a implementação de modelos de regressão, com interface web para facilitar o uso prático por engenheiros e técnicos da indústria de bombeamento.

---

# Apresentação Preliminar

A proposta inicial deste projeto foi apresentada e validada academicamente no:

**XV Congresso de Iniciação Científica e Extensão da Católica de Santa Catarina – Unidade Jaraguá do Sul**

- **Data**: 29 de outubro de 2025 (quarta-feira)
- **Período**: Noturno
- **Local**: Bloco G, Sala G202
- **Horário**: 21h45 às 22h00

A apresentação no congresso substituiu o processo tradicional de aprovação por três professores avaliadores. Durante o evento, o conceito e a viabilidade técnica da proposta foram validados, com destaque para a metodologia de Machine Learning aplicada à otimização de ajuste de rotores em bombas centrífugas. A aprovação recebida pelos professores avaliadores presentes reforçou a relevância industrial e científica do projeto, autorizando o prosseguimento.

---

## 1. Introdução

### 1.1. Contexto

A indústria de bombeamento utiliza há décadas fórmulas empíricas de proporcionalidade para ajustar o diâmetro de rotores em bombas centrífugas, buscando adequar a performance da bomba às condições operacionais desejadas. Essas fórmulas relacionam vazão (Q), pressão (H) e potência (N) ao diâmetro do rotor (D) através de relações matemáticas simples:

```math
\frac{Q}{Q_1} = \frac{D}{D_1} \qquad
\frac{H}{H_1} = \left(\frac{D}{D_1}\right)^2 \qquad
\frac{N}{N_1} = \left(\frac{D}{D_1}\right)^3
```

Embora funcionais como ponto de partida, essas equações apresentam imprecisões significativas quando confrontadas com dados reais de operação. A lacuna entre o desempenho projetado e o resultado efetivo gera perdas de eficiência, aumento de custos operacionais e desperdício energético.

### 1.2. Justificativa

Com o avanço das tecnologias de Inteligência Artificial e a disponibilidade crescente de dados históricos de operação, torna-se viável desenvolver modelos preditivos que superem as limitações das abordagens tradicionais. A aplicação de Machine Learning neste contexto representa:

- **Relevância Industrial**: Impacto direto na eficiência energética e redução de custos operacionais em diversos setores (saneamento, indústria química, mineração, agricultura).
- **Inovação Tecnológica**: Modernização de processos de engenharia tradicionalmente baseados em métodos empíricos.
- **Aplicabilidade Prática**: Solução que pode ser integrada ao fluxo de trabalho de engenheiros e técnicos.

### 1.3. Objetivos

**Objetivo Principal:**
Desenvolver um sistema baseado em Machine Learning capaz de prever coeficientes de ajuste otimizados para rotores de bombas centrífugas, superando a precisão das fórmulas empíricas tradicionais.

**Objetivos Específicos:**
1. Coletar e estruturar dados históricos de operação de bombas centrífugas (curvas características, parâmetros operacionais).
2. Implementar e comparar diferentes modelos de regressão (Random Forest, Redes Neurais, Gradient Boosting, SVR).
3. Criar um pipeline completo de IA com etapas de pré-processamento, treinamento, validação e predição.
4. Desenvolver uma interface web intuitiva para engenheiros realizarem predições de ajuste.
5. Validar os resultados do modelo com dados reais de operação e comparar com as fórmulas tradicionais.
6. Documentar métricas de performance (RMSE, MAE, R²) e análise de viabilidade prática.

---

## 2. Descrição do Projeto

### 2.1. Linha de Projeto
**Projetos com Inteligência Artificial (IA)**

### 2.2. Tema do Projeto
Sistema web de predição de coeficientes de ajuste para rotores de bombas centrífugas utilizando modelos de Machine Learning treinados com dados históricos de operação e curvas características reais.

### 2.3. Propósito e Uso Prático

**Contexto da Aplicação:**
Este projeto será desenvolvido especificamente para uso interno da **Famac Indústria de Máquinas Ltda**, fabricante brasileira de bombas centrífugas. A aplicação utilizará dados históricos da própria empresa e será restrita aos seus engenheiros e técnicos.

**Problema Resolvido:**
As fórmulas empíricas atuais geram uma lacuna de precisão entre o desempenho teórico e real das bombas após o ajuste do rotor. Isso resulta em:
- Seleção inadequada de bombas para aplicações específicas dos clientes da Famac
- Custos operacionais elevados para clientes finais
- Necessidade de múltiplos ajustes até encontrar o ponto ótimo
- Retrabalho e desperdício de material no processo de usinagem de rotores

**Uso Prático:**
O sistema permitirá que engenheiros e técnicos da Famac:
1. Insiram os parâmetros da bomba atual (diâmetro original, vazão, pressão, rotação).
2. Especifiquem as condições operacionais desejadas pelo cliente.
3. Recebam a predição do diâmetro ideal do rotor com maior precisão.
4. Visualizem comparações entre o método tradicional e a predição do modelo.
5. Acessem métricas de confiabilidade da predição.
6. Otimizem o processo de manufatura e reduzam custos de produção.

### 2.4. Público-Alvo

- **Primário**: Engenheiros mecânicos, engenheiros de processos e técnicos da **Famac Indústria de Máquinas Ltda** envolvidos no dimensionamento, ajuste e manufatura de bombas centrífugas.
- **Secundário**: Equipe de suporte técnico e vendas da Famac que assessoram clientes.
- **Beneficiários Indiretos**: Clientes finais da Famac nos setores de saneamento básico, indústria química e petroquímica, mineração, agricultura (irrigação) e geração de energia.

**Observação:** A aplicação será de uso **interno e exclusivo** da Famac, não sendo disponibilizada publicamente.

### 2.5. Problemas a Resolver

1. **Imprecisão das Fórmulas Empíricas**: Superação das limitações das equações de proporcionalidade tradicionais.
2. **Falta de Ferramentas Digitais**: Ausência de sistemas modernos que integrem dados históricos para predição.
3. **Desperdício Energético**: Redução de perdas por operação fora do ponto ótimo.
4. **Tempo de Ajuste**: Diminuição do tempo necessário para encontrar o ajuste ideal.
5. **Acesso a Conhecimento**: Democratização de expertise através de um sistema inteligente acessível.

### 2.6. Diferenciação/Ineditismo

**Diferenciação em relação a soluções existentes:**
- **Abordagem Data-Driven Proprietária**: Substituição de fórmulas genéricas por modelos treinados com dados históricos reais da Famac, refletindo as características específicas dos produtos da empresa.
- **Especificidade de Domínio**: Foco exclusivo em bombas centrífugas fabricadas pela Famac, considerando suas linhas de produtos e especificidades construtivas.
- **Pipeline Completo**: Integração de coleta de dados internos, modelagem, validação e interface de usuário adaptada ao fluxo de trabalho da Famac.
- **Interpretabilidade**: Comparação explícita entre método tradicional e predição ML, com métricas de confiança, facilitando a adoção pelos engenheiros da empresa.
- **Vantagem Competitiva**: Sistema proprietário que diferencia a Famac no mercado, agregando valor tecnológico ao processo de manufatura e suporte técnico.
- **Aprendizado Contínuo**: Capacidade de melhorar com feedback de cada novo projeto executado pela empresa.

### 2.7. Limitações

O projeto **NÃO** abrangerá:
- Bombas de outros tipos (deslocamento positivo, bombas axiais, bombas de vácuo).
- Bombas de fabricantes concorrentes (apenas linha de produtos Famac).
- Modificações estruturais no rotor além do ajuste de diâmetro.
- Análise de desgaste e manutenção preditiva (fora do escopo inicial).
- Simulação fluidodinâmica (CFD) dos rotores.
- Controle automático de bombas em tempo real.
- Integração com sistemas SCADA ou controladores industriais (pode ser evolução futura).
- Disponibilização pública ou licenciamento para terceiros (sistema de uso interno exclusivo da Famac).

### 2.8. Normas e Legislações Aplicáveis

| Norma/Legislação | Aplicação no Projeto |
|------------------|----------------------|
| **LGPD (Lei nº 13.709/2018)** | Proteção de dados operacionais e informações de clientes da Famac. Coleta mínima de informações, anonimização de dados de clientes finais, política de privacidade interna, segurança no armazenamento de dados proprietários. |
| **ISO/IEC 27001** | Segurança da informação para armazenamento de dados históricos proprietários da Famac e proteção do modelo treinado (ativo intelectual da empresa). |
| **Licenças de Software (MIT, Apache, BSD)** | Uso adequado de bibliotecas open-source (scikit-learn, TensorFlow, PyTorch, pandas, NumPy). Atribuição de créditos conforme licenciamento. |
| **OWASP Top 10** | Segurança da aplicação web interna (proteção contra injeção, XSS, autenticação fraca), restringindo acesso apenas a colaboradores autorizados da Famac. |
| **OECD AI Principles** | Uso responsável de IA, transparência nas predições, explicabilidade do modelo para engenheiros internos. |
| **Propriedade Intelectual** | O modelo treinado e os dados utilizados são propriedade exclusiva da Famac Indústria de Máquinas Ltda. Confidencialidade de informações técnicas e comerciais. |

### 2.9. Métricas de Sucesso

| Métrica | Critério de Sucesso |
|---------|---------------------|
| **Precisão do Modelo (R²)** | R² > 0.90 na predição de vazão e pressão após ajuste |
| **Erro Médio Absoluto (MAE)** | MAE < 5% em relação aos valores reais de operação |
| **Erro Quadrático Médio (RMSE)** | RMSE inferior ao erro médio das fórmulas empíricas |
| **Tempo de Predição** | Resposta < 2 segundos para uma predição |
| **Taxa de Acerto Prático** | Ajuste recomendado dentro de ±3% do ideal em testes de validação com produtos Famac |
| **Redução de Retrabalho** | Diminuição de 30% no tempo de ajuste fino de rotores na manufatura |
| **Usabilidade (SUS - System Usability Scale)** | Score SUS > 70 em testes com engenheiros da Famac |
| **Adoção Interna** | 80% dos engenheiros utilizando o sistema regularmente após 3 meses |

---

## 3. Especificação Técnica

### 3.1. Requisitos de Software

#### 3.1.1. Requisitos Funcionais (RF)

| ID | Requisito | Prioridade |
|----|-----------|------------|
| RF01 | O sistema deve permitir o cadastro de dados de bombas da linha Famac (modelo, diâmetro original, curva característica) | Alta |
| RF02 | O sistema deve permitir a inserção de parâmetros operacionais (vazão, pressão, rotação) | Alta |
| RF03 | O sistema deve predizer o diâmetro ideal do rotor para condições especificadas | Alta |
| RF04 | O sistema deve apresentar comparação entre fórmula tradicional e predição ML | Alta |
| RF05 | O sistema deve exibir métricas de confiabilidade da predição (intervalo de confiança) | Média |
| RF06 | O sistema deve gerar visualizações de curvas características (original vs. ajustada) | Média |
| RF07 | O sistema deve permitir exportação de resultados (PDF, CSV) para documentação de projetos | Média |
| RF08 | O sistema deve manter histórico de predições realizadas por usuário e projeto | Média |
| RF09 | O sistema deve suportar atualização do modelo com novos dados validados de projetos executados | Baixa |
| RF10 | O sistema deve restringir acesso apenas a colaboradores autorizados da Famac | Alta |
| RF11 | O sistema deve permitir busca de projetos anteriores similares | Baixa |

#### 3.1.2. Requisitos Não-Funcionais (RNF)

| ID | Requisito | Categoria |
|----|-----------|-----------|
| RNF01 | O sistema deve ter tempo de resposta < 2s para predições | Performance |
| RNF02 | O sistema deve suportar pelo menos 10 acessos simultâneos | Escalabilidade |
| RNF03 | O modelo deve ter acurácia mínima de R² = 0.90 | Qualidade |
| RNF04 | A interface deve ser responsiva (mobile-friendly) | Usabilidade |
| RNF05 | O sistema deve ter disponibilidade de 99% | Disponibilidade |
| RNF06 | Dados de treinamento (propriedade da Famac) devem ser armazenados de forma segura e criptografada | Segurança |
| RNF07 | O código deve ter cobertura de testes >= 80% | Manutenibilidade |
| RNF08 | O sistema deve seguir padrões WCAG 2.1 nível AA | Acessibilidade |
| RNF09 | O sistema deve ter logs auditáveis de predições e acessos para controle interno | Rastreabilidade |
| RNF10 | O modelo deve ser versionado e permitir rollback | Manutenibilidade |
| RNF11 | O sistema deve operar em rede local da Famac (intranet) com acesso VPN opcional | Segurança |
| RNF12 | Dados proprietários não devem ser expostos a serviços externos de terceiros | Confidencialidade |

#### 3.1.3. Representação dos Requisitos - Diagrama de Casos de Uso

```
                    ┌───────────────────────────────────────┐
                    │   Sistema de Otimização de Rotores   │
                    │         (Famac - Uso Interno)         │
                    └───────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
    ┌───┴──────┐                ┌───┴────┐                 ┌────┴────┐
    │Engenheiro│                │Técnico │                 │  Admin  │
    │  Famac   │                │ Famac  │                 │ Sistema │
    └───┬───── ┘                └───┬────┘                 └────┬────┘
        │                           │                           │
        │                           │                           │
        ├─► [Cadastrar Bomba]       │                           │
        │                           │                           │
        ├─► [Inserir Parâmetros] ◄──┤                           │
        │          Operacionais     │                           │
        │                           │                           │
        ├─► [Solicitar Predição] ◄──┤                           │
        │          de Ajuste        │                           │
        │                           │                           │
        ├─► [Visualizar Comparação]─┤                           │
        │          Métodos          │                           │
        │                           │                           │
        ├─► [Gerar Relatório]    ◄──┤                           │
        │                           │                           │
        │                           ├─► [Fornecer Feedback]     │
        │                           │      de Precisão          │
        │                           │                           │
        │                           │                           ├─► [Atualizar Modelo]
        │                           │                           │
        │                           │                           ├─► [Gerenciar Dados]
        │                           │                           │      de Treinamento
        │                           │                           │
        │                           │                           ├─► [Monitorar Performance]
        │                           │                           │      do Sistema
```

#### 3.1.4. Aderência aos Requisitos da Linha de Projeto (IA)

| Requisito Obrigatório | Como será atendido |
|-----------------------|-------------------|
| **Aplicar abordagem de IA fundamentada** | Uso de modelos de ML/DL (Random Forest, Redes Neurais, Gradient Boosting) para regressão multi-output |
| **Propósito prático e funcional** | Integração em aplicação web com interface para engenheiros, resolvendo problema real da indústria |
| **Utilizar base de dados com justificativa** | Dados reais de curvas características + dados sintéticos gerados por simulação física quando necessário |
| **Pipeline funcional completo** | Extração de dados → Pré-processamento (normalização, feature engineering) → Treinamento → Validação → Deploy |
| **Responsabilidade ética** | Conformidade com LGPD, transparência nas predições, explicabilidade do modelo |

---

### 3.2. Considerações de Design

#### 3.2.1. Visão Inicial da Arquitetura

O sistema adotará uma arquitetura em três camadas (3-tier) com separação clara entre apresentação, lógica de negócio e dados:

**Componentes Principais:**
1. **Frontend (Camada de Apresentação)**
   - Interface web responsiva
   - Formulários de entrada de dados
   - Visualizações interativas (gráficos de curvas características)

2. **Backend (Camada de Aplicação)**
   - API RESTful para comunicação com frontend
   - Orquestração do pipeline de ML
   - Validação de dados de entrada

3. **Camada de ML/IA**
   - Serviço de predição com modelos treinados
   - Pipeline de pré-processamento
   - Sistema de versionamento de modelos

4. **Camada de Dados**
   - Banco de dados para dados históricos
   - Armazenamento de modelos treinados
   - Cache para predições frequentes

#### 3.2.2. Padrões de Arquitetura

- **Padrão Principal**: Arquitetura em Camadas (Layered Architecture)
- **Padrão de API**: REST (Representational State Transfer)
- **Padrão de ML**: Model-View-Controller adaptado para ML (Model-API-Service)
- **Padrão de Dados**: Repository Pattern para acesso a dados

#### 3.2.3. Modelos C4

**Nível 1 - Contexto do Sistema**

```
┌─────────────────┐                                    ┌──────────────────┐
│   Engenheiro/   │────── Acessa via navegador ────────│   Sistema de     │
│    Técnico      │       para obter predições         │   Otimização     │
└─────────────────┘                                    │   de Rotores     │
                                                       └────────┬─────────┘
                                                                │
                                                                │ Consome dados
                                                                │ de treinamento
                                                                │
                                                       ┌────────▼─────────┐
                                                       │  Base de Dados   │
                                                       │  de Curvas de    │
                                                       │     Bombas       │
                                                       └──────────────────┘
```

**Nível 2 - Containers**

```
┌────────────────────────────────────────────────────────────────┐
│                    Sistema de Otimização de Rotores            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────┐         ┌──────────────────┐              │
│  │   Frontend      │  HTTP   │   Backend API    │              │
│  │   (React/Vue)   │◄────────│   (FastAPI)      │              │
│  │                 │         │                  │              │
│  └─────────────────┘         └────────┬─────────┘              │
│                                       │                        │
│                              ┌────────▼─────────┐              │
│                              │  ML Service      │              │
│                              │  (scikit-learn/  │              │
│                              │   TensorFlow)    │              │
│                              └────────┬─────────┘              │
│                                       │                        │
│                        ┌──────────────┴──────────────┐         │
│                        │                             │         │
│              ┌─────────▼──────┐           ┌──────────▼──────┐  │
│              │   PostgreSQL   │           │  Model Storage  │  │
│              │   (Dados)      │           │   (MLflow/S3)   │  │
│              └────────────────┘           └─────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**Nível 3 - Componentes (Backend API)**

```
┌─────────────────────────────────────────────────────┐
│                    Backend API (FastAPI)            │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────┐      ┌─────────────────┐       │
│  │  API Endpoints  │      │  Auth Service   │       │
│  │  /predict       │      │  (JWT)          │       │
│  │  /bombs         │      └─────────────────┘       │
│  │  /history       │                                │
│  └────────┬────────┘                                │
│           │                                         │
│  ┌────────▼─────────┐      ┌─────────────────┐      │
│  │  Business Logic  │      │  Validation     │      │
│  │  - Orchestration │      │  Layer          │      │
│  │  - Calculations  │      └─────────────────┘      │
│  └────────┬─────────┘                               │
│           │                                         │
│  ┌────────▼─────────┐      ┌─────────────────┐      │
│  │  ML Pipeline     │      │  Data Access    │      │
│  │  - Preprocess    │      │  Layer (DAL)    │      │
│  │  - Predict       │      │  - Repository   │      │
│  │  - Postprocess   │      └─────────────────┘      │
│  └──────────────────┘                               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

#### 3.2.4. Mockups das Telas Principais

**Tela 1: Dashboard Principal**
```
┌───────────────────────────────────────────────────────────────┐
│  Sistema de Otimização de Rotores                  [Usuario]  │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────────────────────────────────────────────┐   │
│  │        Nova Predição de Ajuste                         │   │
│  │                                                        │   │
│  │  Dados da Bomba Atual                                  │   │
│  │  ┌─────────────────────┐  ┌──────────────────────┐     │   │
│  │  │ Modelo: [________]  │  │ Diâmetro: [___] mm   │     │   │
│  │  └─────────────────────┘  └──────────────────────┘     │   │
│  │                                                        │   │
│  │  Parâmetros Operacionais Atuais                        │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │   │
│  │  │Vazão: [____] │ │Pressão:[____]│ │Rotação:[____]│    │   │
│  │  │    m³/h      │ │     mca      │ │     rpm      │    │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘    │   │
│  │                                                        │   │
│  │  Condições Operacionais Desejadas                      │   │
│  │  ┌──────────────┐ ┌──────────────┐                     │   │
│  │  │Vazão: [____] │ │Pressão:[____]│                     │   │
│  │  │    m³/h      │ │     mca      │                     │   │
│  │  └──────────────┘ └──────────────┘                     │   │
│  │                                                        │   │
│  │                [ Calcular Ajuste Ideal ]               │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                               │
│  Histórico de Predições Recentes                              │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Data     │ Modelo  │ D.Original │ D.Sugerido │ Método   │  │
│  ├──────────┼─────────┼────────────┼────────────┼──────────┤  │
│  │10/11/25  │ FSG-2   │  135mm     │  128mm     │ Neural   │  │
│  │09/11/25  │ FSG-2   │  120mm     │  115mm     │ R.Forest │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

**Tela 2: Resultado da Predição**
```
┌────────────────────────────────────────────────────────────────┐
│  ← Voltar              Resultado da Predição                   │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Ajuste Recomendado                                      │  │
│  │                                                          │  │
│  │         Diâmetro Original: 135 mm                        │  │
│  │                                                          │  │
│  │         ╔══════════════════════════════╗                 │  │
│  │         ║  Diâmetro Ideal: 128.3 mm   ║                  │  │
│  │         ╚══════════════════════════════╝                 │  │
│  │                                                          │  │
│  │         Redução: 6.7 mm (4.96%)                          │  │
│  │         Confiança: 94.2% (R² = 0.942)                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Comparação de Métodos                                   │  │
│  │                                                          │  │
│  │  Método Tradicional (Fórmula): 130.5 mm                  │  │
│  │  Modelo ML (Random Forest):    128.3 mm                  │  │
│  │  Diferença:                     2.2 mm (1.7%)            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Curva Característica Projetada                          │  │
│  │                                                          │  │
│  │  Pressão (mca)                                           │  │
│  │    30│                                                   │  │
│  │      │  ●─────●                                          │  │
│  │    25│       ●─────●                                     │  │
│  │      │            ●─────● (Curva Original D=135mm)       │  │
│  │    20│     ○─────○                                       │  │
│  │      │          ○─────○ (Curva Ajustada D=128mm)         │  │
│  │    15│               ○─────○                             │  │
│  │      └──────┬──────┬──────┬──────┬──────                 │  │
│  │             5     10     15     20     25  Vazão (m³/h)  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  [ Exportar PDF ]  [ Nova Predição ]  [ Fornecer Feedback ]    │
└────────────────────────────────────────────────────────────────┘
```

#### 3.2.5. Decisões e Alternativas Consideradas

**Decisão 1: Escolha do Framework Backend**
- **Escolhido**: FastAPI
- **Alternativas**: Flask, Django
- **Justificativa**: Performance superior, suporte nativo a async, documentação automática (OpenAPI), tipagem com Pydantic, ideal para APIs de ML.

**Decisão 2: Modelos de ML a Implementar**
- **Escolhidos**: Random Forest, Redes Neurais (MLP), Gradient Boosting (XGBoost)
- **Alternativas**: SVR, Regressão Linear, KNN
- **Justificativa**: Ensemble methods e redes neurais oferecem melhor balance entre precisão e interpretabilidade para problemas de regressão multi-output.

**Decisão 3: Armazenamento de Dados**
- **Escolhido**: PostgreSQL para dados estruturados + MinIO (on-premises) para modelos
- **Alternativas**: MongoDB, MySQL, SQLite, AWS S3
- **Justificativa**: PostgreSQL oferece robustez, suporte a JSON, escalabilidade e conformidade ACID. MinIO on-premises garante que dados proprietários da Famac permaneçam internos, sem exposição a serviços cloud públicos.

**Decisão 4: Frontend Framework**
- **Escolhido**: React com TypeScript
- **Alternativas**: Vue.js, Angular, Svelte
- **Justificativa**: Ecossistema maduro, biblioteca Chart.js para visualizações, forte tipagem com TypeScript.

#### 3.2.6. Critérios de Escalabilidade, Resiliência e Segurança

**Escalabilidade:**
- Arquitetura stateless permitindo escalamento horizontal
- Cache de predições frequentes (Redis)
- Containerização com Docker para deploy escalável
- Load balancer para distribuição de carga (se necessário no futuro)
- Deploy em servidor on-premises da Famac (ou cloud privada caso a empresa utilize)

**Resiliência:**
- Versionamento de modelos com possibilidade de rollback
- Circuit breaker para falhas no serviço de ML
- Logs estruturados e monitoramento (Prometheus/Grafana)
- Health checks em todos os serviços
- Backup automatizado de dados e modelos

**Segurança:**
- Autenticação via JWT (JSON Web Tokens) integrada com AD/LDAP da Famac (se disponível)
- HTTPS obrigatório em produção
- Validação de input com Pydantic
- Rate limiting para prevenir abuse
- Criptografia de dados proprietários em repouso (AES-256)
- Sanitização de dados de entrada para prevenir injeção
- Acesso restrito à rede interna da Famac (firewall, VPN)
- Controle de acesso baseado em funções (RBAC) por departamento

---

### 3.3. Stack Tecnológica

#### 3.3.1. Linguagens de Programação

| Linguagem | Uso | Justificativa |
|-----------|-----|---------------|
| **Python 3.11+** | Backend, ML Pipeline | Ecossistema robusto de bibliotecas de ML/Data Science, performance, tipagem estática opcional |
| **TypeScript** | Frontend | Type safety, melhor manutenibilidade, IDE support superior |
| **SQL** | Queries e DDL | Manipulação eficiente de dados estruturados |

#### 3.3.2. Frameworks e Bibliotecas

**Backend:**
- **FastAPI**: Framework web assíncrono de alta performance
- **Pydantic**: Validação de dados e serialização
- **SQLAlchemy**: ORM para abstração de banco de dados
- **Alembic**: Migrações de banco de dados

**Machine Learning:**
- **scikit-learn**: Implementação de Random Forest, pré-processamento, métricas
- **TensorFlow/Keras**: Redes Neurais (MLP)
- **XGBoost**: Gradient Boosting otimizado
- **pandas**: Manipulação de datasets
- **NumPy**: Computação numérica
- **matplotlib/seaborn**: Visualizações para análise exploratória

**Frontend:**
- **React 18**: Biblioteca UI component-based
- **Vite**: Build tool e dev server rápido
- **TailwindCSS**: Framework CSS utility-first
- **Chart.js**: Visualizações interativas de gráficos
- **Axios**: Cliente HTTP
- **React Hook Form**: Gerenciamento de formulários

**Infraestrutura e DevOps:**
- **Docker**: Containerização
- **PostgreSQL**: Banco de dados relacional
- **Redis**: Cache e sessions
- **MinIO**: Storage on-premises para modelos (alternativa S3)
- **MLflow**: Versionamento e tracking de modelos
- **Nginx**: Reverse proxy e balanceamento
- **pytest**: Framework de testes (Python)
- **Jest**: Framework de testes (JavaScript)

#### 3.3.3. Ferramentas de Desenvolvimento e Gestão

| Ferramenta | Propósito |
|------------|-----------|
| **VS Code** | IDE principal |
| **Git/GitHub** | Versionamento de código |
| **GitHub Actions** | CI/CD pipeline |
| **Postman/Insomnia** | Teste de APIs |
| **DBeaver** | Gerenciamento de banco de dados |


#### 3.3.4. Licenciamento

| Software/Biblioteca | Licença | Implicações |
|---------------------|---------|-------------|
| FastAPI | MIT | Uso livre, inclusive comercial |
| scikit-learn | BSD 3-Clause | Uso livre, redistribuição permitida |
| TensorFlow | Apache 2.0 | Uso livre, patentes protegidas |
| React | MIT | Uso livre, inclusive comercial |
| PostgreSQL | PostgreSQL License (similar BSD) | Uso livre, open-source |
| Chart.js | MIT | Uso livre, inclusive comercial |
| XGBoost | Apache 2.0 | Uso livre, patentes protegidas |

**Nota sobre Licenciamento**: Todo o código desenvolvido será distribuído sob licença MIT, permitindo uso, modificação e distribuição livre, com atribuição de créditos.

---

### 3.4. Considerações de Segurança

#### 3.4.1. Riscos Identificados

| Risco | Nível | Descrição |
|-------|-------|-----------|
| **Injeção de Dados Maliciosos** | Alto | Input manipulation para comprometer predições ou banco de dados |
| **Vazamento de Dados Proprietários da Famac** | Alto | Acesso não autorizado a curvas de bombas, especificações técnicas e know-how da empresa |
| **Model Poisoning** | Médio | Inserção de dados corrompidos no sistema de feedback |
| **Ataques DDoS** | Baixo | Sobrecarga do serviço (mitigado por estar em rede interna) |
| **Acesso não Autorizado Externo** | Médio | Tentativas de acesso de fora da rede corporativa |
| **Acesso não Autorizado Interno** | Médio | Colaboradores não autorizados acessando o sistema |
| **Perda de Dados por Falha de Hardware** | Médio | Indisponibilidade do sistema por falha em servidor local |
| **XSS (Cross-Site Scripting)** | Médio | Injeção de scripts maliciosos na interface |
| **Exposição de Credenciais** | Alto | Vazamento de secrets, API keys ou tokens |

#### 3.4.2. Medidas de Mitigação

**Prevenção de Injeção:**
- Validação rigorosa de inputs com Pydantic schemas
- Parametrização de queries SQL (uso de ORM)
- Sanitização de dados antes do processamento
- Limites de range para parâmetros numéricos

**Proteção de Dados Proprietários:**
- Criptografia em trânsito (TLS 1.3)
- Criptografia em repouso para dados proprietários da Famac (AES-256)
- Anonimização de dados de clientes finais da Famac
- Política de retenção de dados (LGPD compliance)
- Backup criptografado em storage local da empresa
- Sem transmissão de dados para serviços externos/cloud pública
- Acordo de confidencialidade (NDA) para desenvolvedores com acesso ao sistema

**Controle de Acesso:**
- Autenticação JWT com expiração de tokens (integração com AD/LDAP da Famac)
- RBAC (Role-Based Access Control) - engenheiro, técnico, admin, visualizador
- Restrição de acesso à rede interna/VPN da Famac
- Whitelist de IPs autorizados
- Rate limiting por IP e por usuário
- Logs de auditoria de acessos e ações (conformidade com políticas internas)
- Sessões com timeout automático

**Proteção da Aplicação:**
- Content Security Policy (CSP) headers
- CORS configurado adequadamente
- Proteção contra CSRF (tokens anti-CSRF)
- Sanitização de outputs para prevenir XSS
- Helmet.js para headers de segurança

**Segurança do Modelo:**
- Validação de dados de feedback antes de incorporar ao retreinamento
- Versionamento de modelos com rollback capability
- Monitoramento de drift do modelo
- Isolamento do serviço de ML em container separado

#### 3.4.3. Normas e Boas Práticas Seguidas

**OWASP Top 10 (2021):**
- A01: Broken Access Control → Autenticação JWT + RBAC
- A02: Cryptographic Failures → TLS 1.3 + AES-256
- A03: Injection → Validação Pydantic + ORM
- A04: Insecure Design → Threat modeling na fase de design
- A05: Security Misconfiguration → Hardening de containers, secrets management
- A07: Identification and Authentication Failures → JWT com refresh tokens
- A08: Software and Data Integrity Failures → Versionamento de modelos
- A09: Security Logging and Monitoring Failures → Logs estruturados + alertas

**ISO/IEC 27001:**
- Implementação de controles de segurança da informação
- Classificação de dados (públicos, internos, confidenciais)
- Gestão de incidentes de segurança
- Política de backup e recuperação

**LGPD (Lei nº 13.709/2018):**
- Minimização de coleta de dados pessoais
- Consentimento explícito para uso de dados
- Direito de acesso, correção e exclusão de dados
- Política de privacidade transparente
- DPO (Data Protection Officer) - definir responsável

#### 3.4.4. Responsabilidade Ética

**Transparência e Explicabilidade:**
- Documentação clara sobre como o modelo funciona
- Explicação das features mais importantes (SHAP values)
- Comparação explícita com método tradicional
- Intervalo de confiança nas predições

**Privacidade:**
- Não coletar dados pessoais desnecessários
- Anonimização de dados de clientes/empresas
- Política clara de uso de dados
- Conformidade com LGPD

**Viés e Fairness:**
- Garantir que dados de treinamento sejam representativos
- Monitorar performance do modelo em diferentes cenários
- Documentar limitações conhecidas
- Não discriminar por fabricante ou região

**Uso Responsável:**
- Sistema como ferramenta de apoio à decisão (não substituição total do engenheiro)
- Alertas quando predição está fora de faixa confiável
- Documentação de casos de uso apropriados
- Disclaimer sobre responsabilidade final do engenheiro

---

### 3.5. Conformidade e Normas Aplicáveis

#### Tabela Detalhada de Conformidade

| Norma/Legislação | Aplicação Específica no Projeto | Medidas de Conformidade |
|------------------|----------------------------------|-------------------------|
| **LGPD (Lei nº 13.709/2018)** | Tratamento de dados operacionais e informações de clientes da Famac | • Coleta mínima de dados (apenas parâmetros técnicos necessários)<br>• Política de privacidade interna da Famac<br>• Funcionalidade de exclusão de dados de clientes<br>• Anonimização de dados de clientes finais em datasets<br>• Criptografia de dados em repouso e trânsito<br>• Dados armazenados em infraestrutura da Famac (Brasil) |
| **OWASP Top 10** | Segurança da aplicação web | • Validação de inputs (prevenção de injeção)<br>• Autenticação JWT segura<br>• HTTPS obrigatório<br>• Content Security Policy<br>• Rate limiting<br>• Logs de auditoria |
| **ISO/IEC 27001** | Segurança da informação | • Classificação de dados<br>• Controles de acesso (RBAC)<br>• Gestão de incidentes<br>• Backup criptografado<br>• Política de segurança documentada |
| **OECD AI Principles** | Uso responsável de IA | • Transparência nas predições<br>• Explicabilidade do modelo (SHAP)<br>• Robustez e segurança do sistema<br>• Accountability (logs de decisões)<br>• Human-centered values (ferramenta de apoio) |
| **Licenças de Software (MIT, Apache, BSD)** | Uso de bibliotecas open-source | • Atribuição de créditos em documentação<br>• Arquivo LICENSE no repositório<br>• Compliance com termos de redistribuição<br>• Documentação de dependências (requirements.txt, package.json) |

---

## 4. Próximos Passos

### 4.1. Roadmap de Desenvolvimento

#### Fase 1: Portfólio I (Atual - 2025/1)

**Mês 1-2: Pesquisa e Fundamentação Teórica**
- [x] Estudo de fórmulas empíricas de ajuste de rotores
- [x] Revisão bibliográfica de aplicações de ML em engenharia mecânica
- [x] Análise de trabalhos relacionados
- [x] Preparação de proposta inicial
- [x] Apresentação no XV Congresso de Iniciação Científica da Católica SC (29/10/2025)
- [x] Definição detalhada do escopo técnico
- [x] Especificação completa de requisitos (este RFC)

**Mês 3-4: Coleta e Preparação dos Dados**
- [x] Reunião com equipe técnica da Famac para identificação de fontes de dados
- [x] Coleta de curvas características de bombas da linha Famac (documentação interna)
- [ ] Estruturação do dataset (features, labels)
- [ ] Análise exploratória de dados (EDA)
- [ ] Limpeza e pré-processamento
- [ ] Feature engineering (criação de features derivadas)

**Mês 5-7: Desenvolvimento do Modelo de ML**
- [ ] Implementação do pipeline de pré-processamento
- [ ] Treinamento de modelos baseline (Linear Regression)
- [ ] Implementação de Random Forest Regressor
- [ ] Implementação de Redes Neurais (MLP)
- [ ] Implementação de Gradient Boosting (XGBoost)
- [ ] Comparação de performance dos modelos
- [ ] Otimização de hiperparâmetros (GridSearch/RandomSearch)
- [ ] Validação cruzada (k-fold)

**Mês 8-9: Validação e Análise de Resultados**
- [ ] Testes com dados reais de operação
- [ ] Cálculo de métricas (RMSE, MAE, R², MAPE)
- [ ] Comparação com fórmulas tradicionais
- [ ] Análise de feature importance
- [ ] Explicabilidade do modelo (SHAP values)
- [ ] Documentação de resultados
- [ ] Preparação de artigo/poster de Portfólio I

#### Fase 2: Portfólio II (2025/2)

**Mês 1-2: Desenvolvimento da Interface Web**
- [ ] Setup do projeto frontend (React + TypeScript + Vite)
- [ ] Implementação de componentes de UI
- [ ] Integração com API backend
- [ ] Visualizações de gráficos (Chart.js)
- [ ] Responsividade mobile

**Mês 3-4: Desenvolvimento do Backend**
- [ ] Implementação da API REST (FastAPI)
- [ ] Integração com modelos de ML treinados
- [ ] Setup do banco de dados (PostgreSQL)
- [ ] Implementação de autenticação (JWT)
- [ ] Sistema de cache (Redis)

**Mês 5-6: Testes e Refinamento**
- [ ] Testes unitários (backend e modelos)
- [ ] Testes de integração
- [ ] Testes de performance
- [ ] Testes de usabilidade com engenheiros da Famac (5-10 usuários piloto)
- [ ] Validação com casos reais de projetos anteriores da Famac
- [ ] Correção de bugs e ajustes baseados em feedback interno
- [ ] Otimização de performance

**Mês 7-8: Deployment e Documentação**
- [ ] Containerização com Docker
- [ ] Deploy em cloud (AWS/Azure/GCP)
- [ ] Configuração de CI/CD
- [ ] Monitoramento e logging
- [ ] Documentação técnica completa
- [ ] Manual do usuário

**Mês 9: Finalização**
- [ ] Preparação de apresentação final
- [ ] Elaboração de artigo técnico
- [ ] Criação de vídeo demonstrativo
- [ ] Entrega final de Portfólio II

### 4.2. Definição de Marcos (Milestones)

| Marco | Data Prevista | Entregável |
|-------|---------------|------------|
| **M0: Apresentação e Aprovação no Congresso** | 29/10/2025 ✅ | Proposta aprovada no XV Congresso de Iniciação Científica |
| **M1: RFC Completo** | 30/11/2025 | Documento RFC com especificação técnica detalhada |
| **M2: Dataset Preparado** | 15/01/2026 | Dataset estruturado e validado para treinamento |
| **M3: Modelo Baseline Treinado** | 15/02/2026 | Primeiro modelo funcional com métricas |
| **M4: Modelo Otimizado** | 15/03/2026 | Modelo final com R² > 0.90 |
| **M5: Entrega Portfólio I** | 30/03/2026 | Apresentação de resultados do modelo |
| **M6: Protótipo de Interface** | 30/04/2026 | UI funcional com integração básica |
| **M7: Sistema Completo** | 30/06/2026 | Aplicação web mvp totalmente integrada |

### 4.3. Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| Dados insuficientes para treinamento | Média | Alto | Geração de dados sintéticos + acesso a histórico completo da Famac |
| Modelo com baixa acurácia | Baixa | Alto | Ensemble methods, feature engineering avançado, validação com especialistas da Famac |
| Atraso no desenvolvimento | Média | Médio | Buffer de tempo no cronograma, priorização de features |
| Dificuldades com deployment on-premises | Média | Médio | Levantamento prévio de infraestrutura disponível na Famac, uso de Docker |
| Resistência à adoção interna | Média | Médio | Envolvimento de engenheiros da Famac desde o início, treinamento adequado |
| Restrições de acesso a dados proprietários | Baixa | Alto | NDA assinado, comprometimento de confidencialidade, código-fonte de propriedade da Famac |

---

## 5. Referências

### Artigos e Livros

1. **Karassik, I. J., et al.** (2008). *Pump Handbook*. 4th Edition. McGraw-Hill.
2. **Gülich, J. F.** (2020). *Centrifugal Pumps*. 4th Edition. Springer.
3. **Géron, A.** (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. 2nd Edition. O'Reilly Media.
4. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer.
5. **Hastie, T., Tibshirani, R., Friedman, J.** (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2nd Edition. Springer.

### Frameworks e Bibliotecas

6. **FastAPI Documentation**. [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
7. **scikit-learn Documentation**. [https://scikit-learn.org/](https://scikit-learn.org/)
8. **TensorFlow Documentation**. [https://www.tensorflow.org/](https://www.tensorflow.org/)
9. **React Documentation**. [https://react.dev/](https://react.dev/)
10. **XGBoost Documentation**. [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
11. **PostgreSQL Documentation**. [https://www.postgresql.org/docs/](https://www.postgresql.org/docs/)

### Normas e Regulamentações

12. **LGPD – Lei Geral de Proteção de Dados Pessoais** (Lei nº 13.709/2018). [https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/L13709.htm](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/L13709.htm)
13. **OWASP Top Ten 2021**. [https://owasp.org/www-project-top-ten/](https://owasp.org/www-project-top-ten/)
14. **ISO/IEC 27001:2013 – Information Security Management**. [https://www.iso.org/isoiec-27001-information-security.html](https://www.iso.org/isoiec-27001-information-security.html)
15. **OECD AI Principles**. [https://oecd.ai/en/ai-principles](https://oecd.ai/en/ai-principles)
16. **UNESCO Recommendation on the Ethics of Artificial Intelligence**. [https://unesdoc.unesco.org/ark:/48223/pf0000380455](https://unesdoc.unesco.org/ark:/48223/pf0000380455)

### Padrões de Arquitetura

17. **C4 Model for Software Architecture**. [https://c4model.com/](https://c4model.com/)
18. **Martin, R. C.** (2017). *Clean Architecture: A Craftsman's Guide to Software Structure and Design*. Prentice Hall.
19. **REST API Design Best Practices**. [https://restfulapi.net/](https://restfulapi.net/)

### Datasets e Benchmarks

20. **UCI Machine Learning Repository**. [https://archive.ics.uci.edu/ml/](https://archive.ics.uci.edu/ml/)
21. **Kaggle Datasets**. [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)

### Ferramentas de ML e MLOps

22. **MLflow Documentation**. [https://mlflow.org/docs/latest/index.html](https://mlflow.org/docs/latest/index.html)
23. **SHAP (SHapley Additive exPlanations)**. [https://github.com/slundberg/shap](https://github.com/slundberg/shap)

---

## 6. Apêndices

### Apêndice A: Glossário Técnico

| Termo | Definição |
|-------|-----------|
| **Bomba Centrífuga** | Máquina hidráulica que converte energia mecânica em energia hidráulica por meio da força centrífuga |
| **Rotor/Impulsor** | Componente rotativo da bomba que transfere energia ao fluido |
| **Curva Característica** | Gráfico que relaciona vazão, pressão e eficiência de uma bomba |
| **mca (metros de coluna d'água)** | Unidade de medida de pressão/altura manométrica |
| **Pipeline de ML** | Sequência de etapas de processamento de dados e modelagem |
| **Feature Engineering** | Processo de criar novas variáveis derivadas para melhorar o modelo |
| **Ensemble Methods** | Técnicas que combinam múltiplos modelos para melhor performance |
| **R² (Coeficiente de Determinação)** | Métrica que indica a proporção da variância explicada pelo modelo (0 a 1) |
| **RMSE (Root Mean Square Error)** | Raiz do erro quadrático médio entre predições e valores reais |
| **MAE (Mean Absolute Error)** | Média dos erros absolutos entre predições e valores reais |

### Apêndice B: Fórmulas Empíricas Detalhadas

**Leis de Afinidade para Bombas Centrífugas:**

Para variação de diâmetro do rotor (rotação constante):

```math
\frac{Q_2}{Q_1} = \frac{D_2}{D_1}
```

```math
\frac{H_2}{H_1} = \left(\frac{D_2}{D_1}\right)^2
```

```math
\frac{P_2}{P_1} = \left(\frac{D_2}{D_1}\right)^3
```

Onde:
- Q = Vazão (m³/h)
- H = Altura manométrica (mca)
- P = Potência (kW)
- D = Diâmetro do rotor (mm)
- Subscrito 1 = Condição original
- Subscrito 2 = Condição após ajuste

**Limitações Conhecidas:**
- Válidas apenas para pequenas variações de diâmetro (< 10%)
- Não consideram alterações na eficiência da bomba
- Assumem similaridade geométrica perfeita
- Desprezam efeitos de Reynolds e rugosidade

### Apêndice C: Exemplo de Features do Dataset

| Feature | Tipo | Descrição | Exemplo |
|---------|------|-----------|---------|
| `diameter_original` | Float | Diâmetro original do rotor (mm) | 135.0 |
| `flow_original` | Float | Vazão no ponto de operação original (m³/h) | 15.5 |
| `head_original` | Float | Pressão no ponto de operação original (mca) | 24.8 |
| `rpm` | Integer | Rotação da bomba (rpm) | 1750 |
| `efficiency_original` | Float | Eficiência no ponto original (%) | 78.5 |
| `specific_speed` | Float | Rotação específica (adimensional) | 32.4 |
| `fluid_viscosity` | Float | Viscosidade do fluido (cP) | 1.0 |
| `diameter_adjusted` | Float | **[TARGET]** Diâmetro ajustado (mm) | 128.3 |
| `flow_desired` | Float | Vazão desejada (m³/h) | 12.0 |
| `head_desired` | Float | Pressão desejada (mca) | 18.2 |

---

## 7. Registro de Aprovação

### Aprovação Obtida no XV Congresso de Iniciação Científica

**Data da Aprovação:** 29 de outubro de 2025

**Evento:** XV Congresso de Iniciação Científica e Extensão da Católica de Santa Catarina – Unidade Jaraguá do Sul

**Status:** ✅ **PROJETO APROVADO**

A proposta foi apresentada e aprovada pela banca de professores avaliadores durante o congresso, substituindo o processo tradicional de aprovação individual por três professores. A validação acadêmica obtida no evento autoriza o prosseguimento do projeto conforme especificação técnica apresentada neste RFC.

**Data de atualização do RFC:** 12 de novembro de 2025

**Versão:** 1.0
