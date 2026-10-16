import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Stocks Momentum — Ranking de momentum de ações

    /// admonition | Em construção
    Este notebook ainda está sendo escrito. A ideia é estudar **momentum de preço** em ações da B3:
    retornos acumulados em janelas (ex.: 12-1 meses), ranking cross-sectional e a estabilidade
    desse ranking ao longo do tempo.
    ///

    Conteúdo educacional — não é recomendação de investimento.

    **Author:** Marcelo Wizenberg
    """
    )
    return


if __name__ == "__main__":
    app.run()
