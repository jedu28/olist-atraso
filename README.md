# olist-atraso

Previsão de atraso na entrega de pedidos do dataset público da Olist.

Classificação binária: o pedido vai chegar depois da data estimada?
Só usa informações disponíveis no momento da compra (sem vazamento de dados
sobre a entrega em si).

## Fases

1. Dados e baseline (regressão logística, split temporal)
2. Modelagem séria (LightGBM/XGBoost + MLflow)
3. Código de produção (src/, testes, API FastAPI)
4. Docker + deploy manual na Lambda
5. Automação (GitHub Actions, OIDC)
6. Monitoramento (logs + drift com Evidently)

## Setup

```bash
make setup
```

Baixe os CSVs do [dataset Olist no Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
e coloque em `data/raw/`.

## Rodar a API localmente

```bash
make api
```
