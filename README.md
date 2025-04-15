# Hitchhiker
A few weeks ago, I came across the [DeepLOB](https://arxiv.org/pdf/1808.03668) article. I found it quite interesting, especially if you can find have at hand a rich dataset! My main question was, can something simpler be done in Crypto? The reasons were two-fold:
* The original DeepLOB has over 100K parameters to predict UP, STATIONARY, DOWN movements, using a rich LOB dataset of over 130M rows. Can the ideas be transfered to a different market?. In Kaggle you can find a nice 10 Level LOB (BTCUSD) [Binance-LOB](https://www.kaggle.com/datasets/siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data) recoding 12 consecutive days, starting from January 9th, 2023, until January 20th, 2023,
* Given the reduction in magnitude of freely-available data, can a 10x simpler model for the Crypto Market be created? What about interpretability? 

## LOB dataset
Given my limitations on GPU, I decided to downsample the dataset through (500ms) average periods. To goal was simple, have 0.5 (s) data points. The dataset was nice enough to only require a backfilling to endup with 0 NaNs per feature. 

### Computing Moves (k)
As done in the [DeepLOB](https://arxiv.org/pdf/1808.03668) article, the aim is to predict movement directions by computing 
mid prices (L1 Bid/Ask mid price) and averaged mid prices over past and future $k$ periods. The definition are given below:

*Mid-Price Calculation:* The mid-price ($p_t$, at time $t$) is calculated as the average of the best ask price ($p_{ask\_l1}$) and 
the best bid price ($p_{bid\_l1}$):

$$p_t = \frac{p_{ask\_l1} + p_{bid\_l1}}{2}$$

*Moment Calculation:* Two moments are calculated for each time $t$. The average mid-price over the past (future) $k$ periods:

$$moment\_minus = \frac{1}{k} \sum_{i=t-k}^{t-1} p_i,\quad moment\_plus = \frac{1}{k} \sum_{i=t+1}^{t+k} p_i$$

> [!NOTE] 
> *Handling Out-of-Bound Prices* If the calculation window extends beyond the available data, the moments are adjusted to the remaining dataset.

*Moment Direction:* The moment direction is computed as the percentage change between `moment_plus` and `moment_minus`:

$$moment\_direction = \frac{moment\_plus - moment\_minus}{moment\_minus}$$

If $moment\_minus$ is 0, $moment\_direction$ is set to 0 to avoid division by zero.

*Move Classification:* The move at time $t$ is classified based on the `moment_direction` and a predefined threshold $\alpha$.

$$
move(t) =
\begin{cases}
  1, & \text{if } moment\_direction > \alpha \\
  -1, & \text{if } moment\_direction < -\alpha \\
  0, & \text{otherwise}
\end{cases}
$$

> [!IMPORTANT]
> Notice that $\alpha$ has serious effects on the dataset imbalance. 
> By ensuring a low value, I managed to get a (okay-ish) balanced one. 

The (almost)-balanced dataset for $\alpha = 1e-5$ is given below.

![deep-model-training-distro](images/deep_model_training_distribution.png)

The movements are nicely captured by the classification function (it can be improved further), as seen in the moment directions out of the threshold $\pm \alpha$.

![deep-model-classification](images/moments_move_snapshot.png)

The processed dataset contains $\approx$ 1.6M rows and 42 features, consisting of 10 levels (Bid/Aks) + l1 mid-price + moment-direction, and a single target column: `moves`.

> [!NOTE]
> In terms of exact sizes, the training set had shape `(1697078, 42)` and its target: `(1697078,)`
> The testing set `(174098, 42)`, and its target: `(174098,)` 


## Simple DeepLOB

I implemented a simplified version of the DeepLOB model (using torch) and trained it using Adam method, learning rate (0.003). An important aspect was to keep the model simple, but also keep enough convolution layers, as they nicely encoded volume-adjusted bids and asks, as well as bid-ask averages. Therefore, each layer had a purpose, which I'll explain.
Each batch includes a time-window of 100 previous data points, to be exploited by the convolution layers. 

> [!NOTE] After some experimentation I quickly realize that linear layers did not help with the recurrent nature of the series, and instead introduced more parameters leading to exploding gradients. 

```mermaid
graph LR
    Input["Input [Batch x 1 x 100 x 42]"]:::input

    subgraph Block1["Conv Block 1"]
        direction TB
        B["Conv2D [1,2] Stride [1,2]"]:::conv
        C["LeakyReLU"]:::activation
        D["BatchNorm2D"]:::batchnorm
        B --> C --> D
    end

    subgraph Block2["Conv Block 2"]
        direction TB
        E["Conv2D [4,1] Stride [2,1]"]:::conv
        F["LeakyReLU"]:::activation
        G["BatchNorm2D"]:::batchnorm
        E --> F --> G
    end

    subgraph Block3["Conv Block 3"]
        direction TB
        H["Conv2D [1,2] Stride [1,1]"]:::conv
        I["LeakyReLU"]:::activation
        J["BatchNorm2D"]:::batchnorm
        H --> I --> J
    end

    subgraph Block4["Conv Block 4"]
        direction TB
        K["Conv2D [4,1] Stride [1,1]"]:::conv
        L["LeakyReLU"]:::activation
        M["BatchNorm2D"]:::batchnorm
        K --> L --> M
    end

    subgraph OutputBlock["Output Block"]
        direction TB
        N["Reshape [Batch x 46 x 160]"]:::reshape
        O["GRU [hidden:10, layers:4, dropout:0.3]"]:::gru
        P["Linear [10,3]"]:::linear
        N --> O --> P
    end

    Input --> Block1
    Block1 --> Block2
    Block2 --> Block3
    Block3 --> Block4
    Block4 --> OutputBlock

    classDef input fill:#2C3E50,stroke:#E74C3C,color:#ECF0F1
    classDef conv fill:#2980B9,stroke:#3498DB,color:#ECF0F1
    classDef activation fill:#27AE60,stroke:#2ECC71,color:#ECF0F1
    classDef batchnorm fill:#8E44AD,stroke:#9B59B6,color:#ECF0F1
    classDef reshape fill:#D35400,stroke:#E67E22,color:#ECF0F1
    classDef gru fill:#C0392B,stroke:#E74C3C,color:#ECF0F1
    classDef linear fill:#16A085,stroke:#1ABC9C,color:#ECF0F1
```

As you can see, the net has blocks as defined below ($i,j,k$ varying per block). The convolution aims at capturing linear relationships between the tabulated dataset, especifically relationships between bid/asks prices, and between temporal values.

```mermaid
graph LR
    subgraph Block1["Conv Block i"]
        direction TB
        B["Conv2D [j, k] Stride [j, k]"]
        C["LeakyReLU"]:::activation
        D["BatchNorm2D"]:::batchnorm
        B --> C --> D
    end

    classDef input fill:#2C3E50,stroke:#E74C3C,color:#ECF0F1
    classDef conv fill:#2980B9,stroke:#3498DB,color:#ECF0F1
    classDef activation fill:#27AE60,stroke:#2ECC71,color:#ECF0F1
    classDef batchnorm fill:#8E44AD,stroke:#9B59B6,color:#ECF0F1
```

Therefore:

* [`Conv Block 1`] Computes implicitly volume-adjusted bids/asks via the kernel `(1, 2)`, stride `(1, 2)`. Notice that, this operation is per row. A leaky activation is used for stability purposes, and normalized with `BatchNorm2D`. I call it `space-adjusted` block.
* [`Conv Block 2`] Computes time-dependent adjusted quantities (from the volume-adjusted bids/asks) via the kernel `(4, 1)`. Here, the kernel will look back four points, lagging 2s of information. I call it `temporal-adjusted` block.
* [`Conv Block 3`] As the previous two blocks provide volume-adjusted and time-adjusted bids/asks, the third block aims at capturing the relationships between bids/asks across the adjusted quantities. This is exactly the purpose of the kernel `1, 2`. As before, for stability reasons, `Leazy` + `BatchNorm2D` is added.
* [`Conv Block 4`] As the third block is capturing purely `space-adjusted` information, a block for `temporal-adjusted` information is added. This is specified through the kernel `(4, 1)`.
* [`Output Block`] The convolution blocks aim at capturing different level of bid/ask volume-adjusted quantities including their fair prices. To capture then nonlinear relationships and potentially exploit existing hidden states, a GRU layer is added here. If you ask why not an LSTM?. The answer is simple. I prefer simplicity and explainability.


The total number of parameters of the model: 7777


## Results
The training was not that easy, I encountered oscillatory behavior on the residual and validation errors after a few epochs, nonetheless, with early stopping at an epoch with low residual, a nice confusion matrix can be obtained.

![deep-model-confusion-matrix](images/confusion_matrix_prediction_distribution.png)

Its associated prediction distribution is skewed, something that requires further study. 

![deep-model-prediction-distro](images/deep_model_prediction_distribution.png)

So far, this is interesting, there are patterns to be exploited, and clearly potential direction predictions in short timestamps.