# nlp_lasso

Esse projeto explora a aplicação da regressão LASSO em uma rede neural MLP (multilayer Perceptron) para selecionar apenas as variáveis essenciais para o modelo, visando melhorar o desempenho da rede.

### Estudantes
* [Ana Carolina Souza](https://github.com/acsouza2398)
* [Livia Tanaka](https://github.com/liviatanaka)

## 1. Replicação do paper
O projeto teve como referência o artigo [Design and Application of a Variable Selection Method for Multilayer Perceptron Neural Network With LASSO](https://ieeexplore.ieee.org/document/7444176). O objetivo era replicar os métodos para alcançar os resultados apresentados no estudo. A estrutura da rede MLP utilizada era composta por três camadas: a de entrada, a de saída e a "oculta", que era ativada por uma função tangente hiperbólica. Essa rede passou pelo seguinte processo:

    1. Treinamento inicial da rede neural com o otimizador Adam;
    2. Introdução da penalidade LASSO;
    3. Determinação do parâmetro de redução (shrinkage paramter) $\lambda$ através do método de validação cruzada k-fold;
    4. Eliminação das variáveis de entrada com peso 0;

    Repetição do processo até que não haja nenhuma variável de peso 0.

Para comparar a replicação, foi utilizado o mesmo dataset do artigo, o [Pumadyn32hn](https://www.cs.utoronto.ca/~delve/data/pumadyn/desc.html). Contudo, foram obtidos resultados diferentes do esperado, com um RMSE maior do que o relatado e um $R^2$ score menor. Ao analisar a implementação, notou-se que os pesos dos parâmetros de entrada estavam sofrendo alterações mínimas, o que inviabilizou o treinamento adequado da rede. Foram realizadas diversas alterações no modelo do artigo em busca de melhoria, porém não foi obtido sucesso em nenhuma delas. Por isso, optou-se por utilizar a implementação da [regularização L1 do PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.l1_unstructured.html) para aplicar a técnica.

A replicação pode ser vista no notebook [paper_recreation.ipynb](paper_recreation.ipynb) e a implementação com o Pytorch está em [new_implementation.ipynb](new_implementation.ipynb).

## 2. Aplicação em NLP

Visando aplicar tal técnica em um contexto de NLP, trocou-se a rede neural em um [projeto de busca de Pokémons baseada em vetores](https://github.com/acsouza2398/pkmncards_scrapper/tree/aps2).

**COMO FOI FEITA UMA COMPARAÇÃO DE EFICIENCIA COM A VERSÃO ANTERIOR?**