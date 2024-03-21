<h1> Code Search Net</h1>
<h2>Data Usage</h2>
<p>The raw data is downloaded from this <a href="https://www.kaggle.com/datasets/omduggineni/codesearchnet/data">link</a></p>
<p>From this dataset we have selected 5 files(docstring + code) which contain code in python language: python_train_0.jsonl, python_train_1.jsonl, python_train_2.jsonl, python_train_3.jsonl, python_train_4.json.These files are present in './dataset/python/python/final/jsonl/train/'.
Now here is how we should store this data in root:<br>
<b>root</b><br>
|______Data<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     |____raw_data<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                |______ python_train_0.jsonl<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                |______<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;               |______ python_train_4.jsonl<br>
</p>
<br>
<b>Note:</b><p> After doing data exploration, we have used langdetect library to detect the language of each row in the data. Now since langdetect is <b>not 100% accurate</b>, so it will might generate different records
everytime it is run on raw data.
Thus we have provided the <a href="https://drive.google.com/file/d/1bLikw_SwcHxvVD9AmcK_E73rQajFqKRk/view?usp=sharing">link</a> to the processed data which the model has used to generate the embeddings, which are further used during query time.<br>
<b>Caution: </b> If you don't use the processed data provided in the link, then you will have to feed the data to model again  to generate the embedding for further during query time, and this process of generating embeddings will take <b>3-4 hours.</b>
Thus we have provided the <a href="https://drive.google.com/file/d/1drjOcMPYLHwW_9sTjmou1lNZah-X2E-L/view?usp=sharing">link of embeddings of model_1</a>,
  <a href="https://drive.google.com/file/d/1NFyxfvNocy_1PCCxc_M4weHL58zUSgfg/view?usp=sharing">link of embeddings of model_2</a>,
  <a href="https://drive.google.com/file/d/1cbDh820oqAD_wVQ3WsSV3N4awnG_njav/view?usp=sharing">link of embeddings of model_3</a>,
  <a href="https://drive.google.com/file/d/1kR70NPqQD0w0fTp9bprrOL8zjYTYuCV7/view?usp=sharing">link of embeddings of model_4</a>,
 the models which has generated the embeddings, which are further used during query time.
After downloading these embeddings  this how we should store this embedding in root:<br>
<b>root</b><br>
|______Models<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     |____embeddings_all_mpnet_base_v2.npy<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     |____embeddings_intfloat_e5_base_v2.npy<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     |____embeddings_multilingual_e5_large_instruct.npy<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     |____embeddings_mixedbread_ai_mxbai_embed_2d_large_v1.npy<br>
</p>

<h2> What we have done</h2>
<p>After generating the embeddings of all model . Now the testing data(/Data/testing_data/query.csv) is present, It has 57 queries present in it,  embeddings model will first convert this query into embeddings and it will try to fetch the top 10 docstring that matches with this query using cosine similarity and get top 10 index of these docstring and it fill try to fetch the top 10 code that can possibly be code for the query . Now this query and top 10 code are passed in the <b>Claude Api</b> and the claude api will generate the response whether the code for this query is in the top 10 code or not. Response from the claide is Yes or No for the query. </p>
<h2>Why different model</h2>
<p>Different model have different embedding vector for a particular query and every model can generate different top code for a query , and these code on passing to claude tells whether code is present in the top 10 code data or not , for a paritcular model some query does not give the answer but the other model can give the answer for the query.
every embedding model has different accuracy for the testing data.
</p>
<h2>Trial of other Embeddings Model /  Limitations</h2>
<p>We had tried to do the other embedings like (Salesforce/SFR-Embedding-Mistral) having embeddings dimension
 of 4096 and token size as 32768 but while loading this embedding model from hugging face into the jupyter notebook
 it gives an error of MPS backend out of memory as this model has size of 14.22 GB 
 and  when i try this embedding model on the google collab by changing the system runtime as T4 GPU , it has given the error 'your session 
 crashed after using the all the available ram'.<br>
 Similary, I had also try to used the (intfloat/e5-mistral-7b-instruct) embedding model with embedding vector 
 size as 4096 and token size is 32768.<br>
Since we are not able to use embeddings model of higher vector size ,I have to work on the medium size embeddings
model with vector size of 1024, 768 </p>

