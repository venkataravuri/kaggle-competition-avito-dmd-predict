
# Kaggle's AVITO Demand Prediction Challenge
Kaggle competition Avito demand prediction challenge

Date cleanup
- replace_na(list(image_top_1 = -1, price = -1)) %T>% 

text data
 str_to_lower(txt) %>%  **** should you do it ****
  str_replace_all("[^[:alpha:]]", " ") %>%
  str_replace_all("\\s+", " ") %>%


## Feature Engineering Tasks

  
- price = log1p(price),

char_count
word_count
word_density
punctuation_count
https://www.kaggle.com/codename007/avito-eda-fe-time-series-dt-visualization

- time features

    # nans in 'image' column are parsed as floats
    df['has_image'] = df.image.apply(lambda image: True if type(image) == unicode else False).astype('bool')
    df.drop(['image'], axis=1, inplace=True)
    
    for col in ['title', 'description'for col in ['title', 'description']:
        df[col + '_length'] = df[col].apply(lambda txt: len(txt) if type(txt) == unicode else 0).astype('uint32')
        df.drop([col], axis=1, inplace=True)]:
        df[col + '_length'] = df[col].apply(lambda txt: len(txt) if type(txt) == unicode else 0).astype('uint32')
        df.drop([col], axis=1, inplace=True)


importance figure


