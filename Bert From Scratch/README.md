# Progress So Far:

## Model:

Assembles all the layer components, includes a MLM layer, a Dense Layer with these params.

> Dense(units=VOCAB_SIZE,input_size=HIDDEN_SIZE),

This is basically the Logits Layer and we apply a softmax on it for the probs.

![Here's an overview of the BERT Model](https://github.com/narayanpdas/Deep-Learning-From-Scratch/blob/main/Bert%20From%20Scratch/assests/Model_overview.png)

## Tokenizer:

- Made as to take 1 sentence(For MLM Task) or 2 sentences(For next sentence Prediciton) as input and provide the appropriate input_ids, input_type_ids, attention_mask.

### REPRESENTATION:

> input_ids = (CLS_token, token1 , token2, token3,....max_token_len)

> input_type_ids =(all zeros for 1 sentence) or (0 0 0 .. 1 1 1...0 0 0)  
>  zeros are for padded tokens if any

> attention_mask = (1 1 1 1 1 ...0 0 0 0)  
> zeros are for padded tokens if any

## Layers Added and Implemented So far:

- Dense Layer(units,input_size): More Optimized and Modular Dense Layer from the Neural Networks repo.

- Embedding Layer (max_token_len,vocab_size,hidden_units): The Starting layer for the model to take in the input_ids and input_type_ids and combine those with the postional_embedding_layer to be used as a sinle input for further down the line.

  ![From the Arxiv's Bert Paper 2018-19](https://github.com/narayanpdas/Deep-Learning-From-Scratch/blob/main/Bert%20From%20Scratch/assests/embedding_layer_representation.png)

- Normalization Layer(hidden_units): The Layer to Normalize Outputs from other Layers for more stable training.

- single-head-attention-layer(head_dim,hidden_size): The Crux of the attention module, more on that later down the line.

- multi-head-attention-layer(head_units,head_dim): An manager class to build multiple single-head-attention-layers to break inputs and stich outputs for training of the modules

- Feed-Forward-Network(hidden_size,factor): The network to ensemble the output from the attention layers and activate it further to better understand relationships between the head-attention units and more room for information processing.

- transformer-encoding-block(head_units,head_dim,hidden_size,factor): The Combination of the multi-head-attention-layer and the Feed-Forward-Network to create a single unit we popularly call the transformer unit.Basically again a manager class to manage both of these networks.

## DataLoader(path)

Made to load data as we move on in the training, a single at a time.

- What it Does is loads the pre-tokenized data from a .parquet file.

> \_\_len\_\_(): Gives the Length of the entire set.

> load_x(idx): Returns the y_train(input_tokens),input_type_ids and attention_mask at that index of the dataset.

> load_y(idx): Returns x_masked or x_train(x_input_ids) and loss_mask used for the backward propagation.
