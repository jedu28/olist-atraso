# %% [markdown]
# # Exploração — atraso de entrega Olist
#
# Script no formato "percent" (abre como notebook no VS Code/Jupytext).
# Roda depois de colocar os CSVs da Olist em `data/raw/`.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from src.features import build_dataset

pd.set_option("display.max_columns", None)

# %%
df = build_dataset()
df.shape

# %%
df["atrasado"].value_counts(normalize=True)

# %% [markdown]
# Dataset desbalanceado — poucos pedidos atrasados. Reforça por que
# acurácia não é boa métrica aqui (um modelo que sempre prevê "no prazo"
# já acerta a maioria). PR-AUC e recall fazem mais sentido.

# %%
df.isna().mean().sort_values(ascending=False).head(10)

# %%
df.groupby("mes_compra")["atrasado"].mean().plot(kind="bar", title="Taxa de atraso por mês da compra")
plt.tight_layout()
plt.show()

# %%
df.groupby("estados_diferentes")["atrasado"].mean()

# %% [markdown]
# Pedidos onde cliente e vendedor estão em estados diferentes tendem a
# atrasar mais — faz sentido como proxy de distância.

# %%
df.groupby("customer_state")["atrasado"].mean().sort_values(ascending=False).head(10)

# %%
df["seller_taxa_atraso_historica"].hist(bins=30)
plt.title("Distribuição da taxa de atraso histórica do vendedor")
plt.show()
