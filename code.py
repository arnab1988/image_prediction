# image_prediction

from io import BytesIO

def func(ResNet50Embedder, BERTIndEmbedder, BERTEmbedder2, text_embeddings, train_df, uploaded_img, text):
  model_img = ResNet50Embedder()
  vectors = train_df.resnet50_v
  data = pd.DataFrame({'image': [x for x in uploaded_img.keys()], 'title': [text]})
  data['resnet_50_vector'] = data['image'].apply(lambda img_path: model_img.embed_image('/content/'+ img_path))
  image_uploaded = Image.open(BytesIO(uploaded[[x for x in uploaded_img.keys()][0]]))
  os.remove(f"./{[x for x in uploaded_img.keys()][0]}")

  print('UPLOADED IMAGE:\n')
  plt.figure(figsize = (3, 3))
  plt.imshow(image_uploaded)
  plt.axis('off')
  plt.title(data['title'].values[0])
  plt.show()


from google.colab import files
uploaded = files.upload()

title = input('Enter title of image: ')


func(ResNet50Embedder, BERTIndEmbedder, BERTEmbedder2, text_embeddings, train_df, uploaded_img = uploaded, text = title)
