# Yelp EDA (sample)

- Reviews loaded: **6,990,124**
- Merged businesses: **150,346**
- Merged users: **1,987,925** (from review sample)

## Helpful (= useful votes) summary
|       |        helpful |
|:------|---------------:|
| count |    6.99012e+06 |
| mean  |    1.18462     |
| std   |    3.25379     |
| min   |    0           |
| 25%   |    0           |
| 50%   |    0           |
| 75%   |    1           |
| max   | 1182           |

## Helpful by City (top 10)
| city          |    mean |   median |      n |
|:--------------|--------:|---------:|-------:|
| Reno          | 1.61918 |        1 | 351565 |
| Edmonton      | 1.52132 |        1 | 101821 |
| Sparks        | 1.47931 |        1 |  73030 |
| Metairie      | 1.36965 |        1 |  64358 |
| Philadelphia  | 1.29709 |        0 | 967530 |
| Tucson        | 1.27573 |        0 | 404867 |
| Tampa         | 1.24942 |        0 | 454881 |
| Saint Louis   | 1.1978  |        0 | 253430 |
| Santa Barbara | 1.17971 |        0 | 269622 |
| Boise         | 1.17793 |        0 | 105363 |

## Helpful by Category (top 10)
| category_list                                            |     mean |   median |     n |
|:---------------------------------------------------------|---------:|---------:|------:|
| ['Food' 'Coffee & Tea']                                  | 1.26662  |        0 | 18108 |
| ['Restaurants' 'American (New)']                         | 1.17119  |        0 | 17624 |
| ['Italian' 'Restaurants']                                | 1.06499  |        0 | 26328 |
| ['Restaurants' 'Italian']                                | 1.05578  |        0 | 27248 |
| ['Event Planning & Services' 'Hotels' 'Hotels & Travel'] | 1.04623  |        0 | 12913 |
| ['Coffee & Tea' 'Food']                                  | 1.04435  |        0 | 17768 |
| ['Restaurants' 'Chinese']                                | 1.0402   |        0 | 24976 |
| ['Hotels' 'Event Planning & Services' 'Hotels & Travel'] | 1.03102  |        0 | 13054 |
| ['American (New)' 'Restaurants']                         | 1.02696  |        0 | 19325 |
| ['Pizza' 'Restaurants']                                  | 0.980513 |        0 | 29507 |

## Spearman correlation (selected numeric features)
|                   |    helpful |      stars |      cool |       funny |   text_len_words |   text_len_chars |   exclaim_count |   question_count |   caps_ratio |   sentiment_vader |   review_count_user |       fans |   average_stars |
|:------------------|-----------:|-----------:|----------:|------------:|-----------------:|-----------------:|----------------:|-----------------:|-------------:|------------------:|--------------------:|-----------:|----------------:|
| helpful           |  1         | -0.116928  | 0.510513  |  0.408007   |         0.336009 |         0.337394 |      0.0322973  |      0.137614    |  -0.0287899  |       0.0728187   |          0.28148    |  0.343609  |      -0.0505771 |
| stars             | -0.116928  |  1         | 0.108531  | -0.116791   |        -0.22877  |        -0.218483 |      0.25352    |     -0.165878    |   0.14814    |       0.517753    |          0.0175373  |  0.0119633 |       0.522972  |
| cool              |  0.510513  |  0.108531  | 1         |  0.422656   |         0.217991 |         0.220541 |      0.0783042  |      0.0794923   |   0.0104674  |       0.215898    |          0.342532   |  0.40022   |       0.0831469 |
| funny             |  0.408007  | -0.116791  | 0.422656  |  1          |         0.21699  |         0.216921 |      0.0133989  |      0.149427    |   0.00789167 |       0.0209776   |          0.223776   |  0.269915  |      -0.0591192 |
| text_len_words    |  0.336009  | -0.22877   | 0.217991  |  0.21699    |         1        |         0.99674  |      0.112484   |      0.271321    |  -0.173908   |       0.296193    |          0.222624   |  0.273471  |      -0.117673  |
| text_len_chars    |  0.337394  | -0.218483  | 0.220541  |  0.216921   |         0.99674  |         1        |      0.116869   |      0.271496    |  -0.174131   |       0.308177    |          0.224502   |  0.274916  |      -0.110967  |
| exclaim_count     |  0.0322973 |  0.25352   | 0.0783042 |  0.0133989  |         0.112484 |         0.116869 |      1          |      0.0713444   |   0.176839   |       0.29104     |         -0.00651155 |  0.0266162 |       0.15627   |
| question_count    |  0.137614  | -0.165878  | 0.0794923 |  0.149427   |         0.271321 |         0.271496 |      0.0713444  |      1           |   0.0367677  |      -0.000524495 |          0.0866323  |  0.102038  |      -0.0996676 |
| caps_ratio        | -0.0287899 |  0.14814   | 0.0104674 |  0.00789167 |        -0.173908 |        -0.174131 |      0.176839   |      0.0367677   |   1          |       0.0271156   |         -0.013198   | -0.0159813 |       0.0718462 |
| sentiment_vader   |  0.0728187 |  0.517753  | 0.215898  |  0.0209776  |         0.296193 |         0.308177 |      0.29104    |     -0.000524495 |   0.0271156  |       1           |          0.215181   |  0.227868  |       0.332894  |
| review_count_user |  0.28148   |  0.0175373 | 0.342532  |  0.223776   |         0.222624 |         0.224502 |     -0.00651155 |      0.0866323   |  -0.013198   |       0.215181    |          1          |  0.803463  |       0.0382    |
| fans              |  0.343609  |  0.0119633 | 0.40022   |  0.269915   |         0.273471 |         0.274916 |      0.0266162  |      0.102038    |  -0.0159813  |       0.227868    |          0.803463   |  1         |       0.0687536 |
| average_stars     | -0.0505771 |  0.522972  | 0.0831469 | -0.0591192  |        -0.117673 |        -0.110967 |      0.15627    |     -0.0996676   |   0.0718462  |       0.332894    |          0.0382     |  0.0687536 |       1         |

## Kind of words (TF-IDF, logistic regression)
Phrases most associated with **helpful > 0**:

- indy
- completely
- ton
- fix
- dr
- running
- ride
- choice
- 75
- pair
- decided
- didnt
- shared
- email
- pound
- future
- rule
- christmas
- simple
- huge

Phrases most associated with **helpful == 0**:

- hamburger
- quick lunch
- atmosphere service
- food love
- squeezed
- responsive
- disappoints
- booking
- keeping
- recommended
- satisfy
- good little
- fish tacos
- don live
- friendly service
- noisy
- time visit
- brûlée
- wait table
- best ve

