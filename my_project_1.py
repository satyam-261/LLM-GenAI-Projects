{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1tYffoB-q7RQJPsw5b7CLWsSXroDYu9cL",
      "authorship_tag": "ABX9TyMKj7DKTUAPUOnzpWul+nyk",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/YashwiP/LLM-GenAI-Projects/blob/main/My_Project_1.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "<h1><center><font size=10>Introduction to LLMs and GenAI</center></font></h1>\n",
        "<h1><center>Mini Project 1 : Basics of NLP: Text Cleaning & Vectorization</center></h1>"
      ],
      "metadata": {
        "id": "eNE77cyA3q8M"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "#\n",
        "This project is about sentiment analysis of the reviews of a product. We are building a machine learning model for the same so the humans involvement is less.\n"
      ],
      "metadata": {
        "id": "N7-5b58tbD-v"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Business Context\n",
        "In today’s fast-paced e-commerce landscape, customer reviews significantly influence product perception and buying decisions. Businesses must actively monitor customer sentiment to extract insights and maintain a competitive edge. Ignoring negative feedback can lead to serious issues, such as:\n",
        "\n",
        "* Customer Churn: Unresolved complaints drive loyal customers away, reducing retention and future revenue.\n",
        "\n",
        "* Reputation Damage: Persistent negative sentiment can erode brand trust and deter new buyers.\n",
        "\n",
        "* Financial Loss: Declining sales and shifting customer preference toward competitors directly impact profitability.\n",
        "\n",
        "Actively tracking and addressing customer sentiment is essential for sustained growth and brand strength."
      ],
      "metadata": {
        "id": "PUY37Tp_y_Gw"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Problem Definition\n",
        "A growing e-commerce platform specializing in electronic gadgets collects customer feedback from product reviews, surveys, and social media. With a 200% increase in their customer base over three years and a recent 25% spike in feedback volume, their manual review process is no longer sustainable.\n",
        "\n",
        "To address this, the company aims to implement an AI-driven solution to automatically classify customer sentiments (positive, negative, or neutral).\n",
        "\n",
        "As a Data Scientist, your task is to analyze the provided customer reviews—along with their labeled sentiments—and build a predictive model for sentiment classification."
      ],
      "metadata": {
        "id": "p5Bi4VqW24y_"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Data Dictionary\n",
        "- *Product ID:* An exclusive identification number for each product\n",
        "\n",
        "- *Product Review*: Insights and opinions shared by customers about the product\n",
        "\n",
        "- *Sentiment*: Sentiment associated with the product review, indicating whether the review expresses a positive, negative, or neutral sentiment"
      ],
      "metadata": {
        "id": "E5Z_d1PE3AQR"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Importing the necessary libraries for this project"
      ],
      "metadata": {
        "id": "RS9l8GMWrW1t"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# to read and manipulate the data\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "pd.set_option('max_colwidth', None)    # setting column to the maximum column width as per the data\n",
        "\n",
        "# to visualise data\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "# to use regular expressions for manipulating text data\n",
        "import re\n",
        "\n",
        "# to load the natural language toolkit\n",
        "import nltk\n",
        "nltk.download('stopwords')    # loading the stopwords\n",
        "nltk.download('wordnet')    # loading the wordnet module that is used in stemming\n",
        "\n",
        "# to remove common stop words\n",
        "from nltk.corpus import stopwords\n",
        "\n",
        "# to perform stemming\n",
        "from nltk.stem.porter import PorterStemmer\n",
        "\n",
        "# to create Bag of Words\n",
        "from sklearn.feature_extraction.text import CountVectorizer\n",
        "\n",
        "# to split data into train and test sets\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "# to build a Random Forest model\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "\n",
        "# to compute metrics to evaluate the model\n",
        "from sklearn import metrics\n",
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
        "\n",
        "# To tune different models\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lbA5nkbD6YSS",
        "outputId": "1a70c4c0-9dd6-412d-9a32-72b54abe9e72"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Unzipping corpora/stopwords.zip.\n",
            "[nltk_data] Downloading package wordnet to /root/nltk_data...\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "<h1><center>Loading Data Set</center></h1>"
      ],
      "metadata": {
        "id": "q1selBf0Xwg0"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#loading the data into a pandas dataframe\n",
        "df= pd.read_csv(\"/content/drive/MyDrive/Intro to LLM and GenAI/Part-1/Project-1/Product_Reviews.csv\")"
      ],
      "metadata": {
        "id": "7K2f2sLDZE9f"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Data Overview**\n",
        "\n",
        "Checking the first 3 rows of the data"
      ],
      "metadata": {
        "id": "v3DoxQS-r2Qg"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.head(3)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 403
        },
        "id": "VGquyaAjZYfj",
        "outputId": "abf2d25d-78d4-4dd7-c015-14e1d7dff8df"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "             Product ID  \\\n",
              "0  AVpe7AsMilAPnD_xQ78G   \n",
              "1  AVpe7AsMilAPnD_xQ78G   \n",
              "2  AVpe7AsMilAPnD_xQ78G   \n",
              "\n",
              "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     Product Review  \\\n",
              "0                                                                                                                                                                                                                                                                                                                                                                          I initially had trouble deciding between the paperwhite and the voyage because reviews more or less said the same thing: the paperwhite is great, but if you have spending money, go for the voyage.Fortunately, I had friends who owned each, so I ended up buying the paperwhite on this basis: both models now have 300 ppi, so the 80 dollar jump turns out pricey the voyage's page press isn't always sensitive, and if you are fine with a specific setting, you don't need auto light adjustment).It's been a week and I am loving my paperwhite, no regrets! The touch screen is receptive and easy to use, and I keep the light at a specific setting regardless of the time of day. (In any case, it's not hard to change the setting either, as you'll only be changing the light level at a certain time of day, not every now and then while reading).Also glad that I went for the international shipping option with Amazon. Extra expense, but delivery was on time, with tracking, and I didnt need to worry about customs, which I may have if I used a third party shipping service.   \n",
              "1  Allow me to preface this with a little history. I am (was) a casual reader who owned a Nook Simple Touch from 2011. I've read the Harry Potter series, Girl with the Dragon Tattoo series, 1984, Brave New World, and a few other key titles. Fair to say my Nook did not get as much use as many others may have gotten from theirs.Fast forward to today. I have had a full week with my new Kindle Paperwhite and I have to admit, I'm in love. Not just with the Kindle, but with reading all over again! Now let me relate this review, love, and reading all back to the Kindle. The investment of 139.00 is in the experience you will receive when you buy a Kindle. You are not simply paying for a screen there is an entire experience included in buying from Amazon.I have been reading The Hunger Games trilogy and shall be moving onto the Divergent series soon after. Here is the thing with the Nook that hindered me for the past 4 years: I was never inspired to pick it up, get it into my hands, and just dive in. There was never that feeling of oh man, reading on this thing is so awesome. However, with my Paperwhite, I now have that feeling! That desire is back and I simply adore my Kindle. If you are considering purchasing one, stop thinking about it simply go for it. After a full week, 3 downloaded books, and a ton of reading, I still have half of my battery left as well.Make yourself happy. Inspire the reader inside of you.   \n",
              "2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            I am enjoying it so far. Great for reading. Had the original Fire since 2012. The Fire used to make my eyes hurt if I read too long. Haven't experienced that with the Paperwhite yet.   \n",
              "\n",
              "  Sentiment  \n",
              "0  POSITIVE  \n",
              "1  POSITIVE  \n",
              "2  POSITIVE  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-5e772fb8-248a-4f64-ad64-cba0631f79c1\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Product ID</th>\n",
              "      <th>Product Review</th>\n",
              "      <th>Sentiment</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>AVpe7AsMilAPnD_xQ78G</td>\n",
              "      <td>I initially had trouble deciding between the paperwhite and the voyage because reviews more or less said the same thing: the paperwhite is great, but if you have spending money, go for the voyage.Fortunately, I had friends who owned each, so I ended up buying the paperwhite on this basis: both models now have 300 ppi, so the 80 dollar jump turns out pricey the voyage's page press isn't always sensitive, and if you are fine with a specific setting, you don't need auto light adjustment).It's been a week and I am loving my paperwhite, no regrets! The touch screen is receptive and easy to use, and I keep the light at a specific setting regardless of the time of day. (In any case, it's not hard to change the setting either, as you'll only be changing the light level at a certain time of day, not every now and then while reading).Also glad that I went for the international shipping option with Amazon. Extra expense, but delivery was on time, with tracking, and I didnt need to worry about customs, which I may have if I used a third party shipping service.</td>\n",
              "      <td>POSITIVE</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>AVpe7AsMilAPnD_xQ78G</td>\n",
              "      <td>Allow me to preface this with a little history. I am (was) a casual reader who owned a Nook Simple Touch from 2011. I've read the Harry Potter series, Girl with the Dragon Tattoo series, 1984, Brave New World, and a few other key titles. Fair to say my Nook did not get as much use as many others may have gotten from theirs.Fast forward to today. I have had a full week with my new Kindle Paperwhite and I have to admit, I'm in love. Not just with the Kindle, but with reading all over again! Now let me relate this review, love, and reading all back to the Kindle. The investment of 139.00 is in the experience you will receive when you buy a Kindle. You are not simply paying for a screen there is an entire experience included in buying from Amazon.I have been reading The Hunger Games trilogy and shall be moving onto the Divergent series soon after. Here is the thing with the Nook that hindered me for the past 4 years: I was never inspired to pick it up, get it into my hands, and just dive in. There was never that feeling of oh man, reading on this thing is so awesome. However, with my Paperwhite, I now have that feeling! That desire is back and I simply adore my Kindle. If you are considering purchasing one, stop thinking about it simply go for it. After a full week, 3 downloaded books, and a ton of reading, I still have half of my battery left as well.Make yourself happy. Inspire the reader inside of you.</td>\n",
              "      <td>POSITIVE</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>AVpe7AsMilAPnD_xQ78G</td>\n",
              "      <td>I am enjoying it so far. Great for reading. Had the original Fire since 2012. The Fire used to make my eyes hurt if I read too long. Haven't experienced that with the Paperwhite yet.</td>\n",
              "      <td>POSITIVE</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-5e772fb8-248a-4f64-ad64-cba0631f79c1')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-5e772fb8-248a-4f64-ad64-cba0631f79c1 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-5e772fb8-248a-4f64-ad64-cba0631f79c1');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    <div id=\"df-0b20f5f4-e708-4675-a653-b045b384fd81\">\n",
              "      <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-0b20f5f4-e708-4675-a653-b045b384fd81')\"\n",
              "                title=\"Suggest charts\"\n",
              "                style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "      </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "      <script>\n",
              "        async function quickchart(key) {\n",
              "          const quickchartButtonEl =\n",
              "            document.querySelector('#' + key + ' button');\n",
              "          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "          quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "          try {\n",
              "            const charts = await google.colab.kernel.invokeFunction(\n",
              "                'suggestCharts', [key], {});\n",
              "          } catch (error) {\n",
              "            console.error('Error during call to suggestCharts:', error);\n",
              "          }\n",
              "          quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "          quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "        }\n",
              "        (() => {\n",
              "          let quickchartButtonEl =\n",
              "            document.querySelector('#df-0b20f5f4-e708-4675-a653-b045b384fd81 button');\n",
              "          quickchartButtonEl.style.display =\n",
              "            google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "        })();\n",
              "      </script>\n",
              "    </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "df",
              "summary": "{\n  \"name\": \"df\",\n  \"rows\": 1007,\n  \"fields\": [\n    {\n      \"column\": \"Product ID\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 66,\n        \"samples\": [\n          \"AVsRinWmQMlgsOJE6zxC\",\n          \"AVzRlqklGV-KLJ3aavB0\",\n          \"AVpe7AsMilAPnD_xQ78G\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Product Review\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 908,\n        \"samples\": [\n          \"Originally got the walnut cover with my Oasis. It's very nice, and I would describe it as a suede material. For me, the walnut cover didn't age well. Lots of scratching, which is' necessarily a bad thing, but it also seemed to darken a bit which made it a bit dirty. The merlot cover is also very nice and ages a whole lot better. After some months of use, it looks as good today as when I first bought it. Oasis covers in general are very functional. The 2nd battery in the cover is genius. Since I found I really like reading my Oasis without the cover, I love the easy on/off design of the case.\",\n          \"Bought the Echo Tap, Echo Dot, Phillips Hue Starter Kit and a Honeywell Thermostat for my wife for her birthday.She loves it and so do I!Cant wait to add more automated products to it. No need to have my phone out fumbling through multiple apps to change lighting and temperature, check weather, play music, etc., I just ask Alexa to do it while I continue at what I'm doing.Also, when the Dot and Tap are in the same room, the one that is closest to you typically answers your requests. Nice device that will only improve with time.\",\n          \"I habe the echo and I bought this one fort my son's room but it's not as user friendly. I wish I would've just bought the other one.\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Sentiment\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"POSITIVE\",\n          \"NEUTRAL\",\n          \"NEGATIVE\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "/content/drive/MyDrive/Intro to LLM and GenAI/Part-1/Product_Reviews.csv\n"
      ],
      "metadata": {
        "id": "BLo6sYpbY6sk"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Checking the shape of the dataset"
      ],
      "metadata": {
        "id": "zQdmiBhxsR0W"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cJaWpzLVbdqp",
        "outputId": "edb27770-ca55-495c-dd9c-a7ec1d1db35d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(1007, 3)"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "The dataset has 1007 rows and 3 columns"
      ],
      "metadata": {
        "id": "B9xOBErpb58K"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Checking for Missing Values"
      ],
      "metadata": {
        "id": "DVTwTZVwsaEA"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 178
        },
        "id": "wxxRc_h6be5R",
        "outputId": "6f4cfcfa-7c6f-4ae3-84d0-8fd9b68c5a1f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Product ID        0\n",
              "Product Review    0\n",
              "Sentiment         0\n",
              "dtype: int64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>Product ID</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Product Review</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Sentiment</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> int64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "The dataset has no missing values.\n"
      ],
      "metadata": {
        "id": "aTI2AI-VcnBF"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Checking for duplicate values**"
      ],
      "metadata": {
        "id": "-MSMdTQ0c-Or"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.duplicated().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hWaf0dETdc4q",
        "outputId": "9276d695-7ca1-4939-9fb6-3e4fe954fcae"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "np.int64(2)"
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "*   There are 2 duplicate values in the dataset\n",
        "*   We'll drop them\n"
      ],
      "metadata": {
        "id": "Yc9Rr3HGsnIr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#dropping duplicate values\n",
        "df = df.drop_duplicates()\n",
        "df.duplicated().sum()"
      ],
      "metadata": {
        "id": "fU3xzXKpdkdv",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "7f17c365-5e81-4e42-8690-6d21d919c77e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "np.int64(0)"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Dropped the duplicate values."
      ],
      "metadata": {
        "id": "zkvYuOE23wIR"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Checking the shape of the modified dataset"
      ],
      "metadata": {
        "id": "az8S3NlytAAV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.shape"
      ],
      "metadata": {
        "id": "n7hETtGegke2",
        "outputId": "586c9f6e-ada6-4eb0-aa5e-d502b21d5763",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(1005, 3)"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "After dropping the duplicate values we got 1005 rows and 3 columns in the dataset"
      ],
      "metadata": {
        "id": "-1SsiS3H35Le"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Exploratory Data Analysis (EDA)**"
      ],
      "metadata": {
        "id": "d-XQsdoneXcA"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Distribution of sentiments."
      ],
      "metadata": {
        "id": "7EWbhMVXekMm"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "plt\n",
        "sns.countplot(data=df, x=\"Sentiment\");"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 449
        },
        "id": "i1CaO-6LeqEK",
        "outputId": "d6a9efb5-1cc6-444e-ae58-368f788860dd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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AABLI+wAAABLI+wAAABLI+wAAABLI+wAAABLI+wAAABLI+wAAABLI+wAAABL82vYefDBB2Wz2XyW7t27m/3Hjh1TZmamoqKiFB4ertTUVJWWlvpso6SkRCkpKWrdurWio6OVnZ2t48ePn+1DAQAAAaqFvwu49NJLtXr1avNzixb/V9LEiRP1z3/+U4sXL5bD4dD48eN1yy23aOPGjZKkmpoapaSkyOVyadOmTdq/f79GjRqlli1b6tFHHz3rxwIAAAKP38NOixYt5HK56rVXVFTo+eef16JFi9S/f39JUn5+vnr06KHNmzerT58+WrlypT7++GOtXr1aTqdTV155pR5++GFNmjRJDz74oEJCQs724QAAgADj93t2du/erZiYGF100UVKS0tTSUmJJKmoqEjV1dVKSkoyx3bv3l1xcXEqLCyUJBUWFiohIUFOp9Mck5ycLK/Xq507d550n5WVlfJ6vT4LAACwJr+Gnd69e2vBggVasWKF5syZo+LiYv3P//yPDh8+LI/Ho5CQEEVERPis43Q65fF4JEkej8cn6NT11/WdTG5urhwOh7nExsY27YEBAICA4dfLWIMHDzZ/ffnll6t3797q2LGjXn31VYWFhTXbfnNycpSVlWV+9nq9BB4AACzK75exThQREaGuXbvq888/l8vlUlVVlcrLy33GlJaWmvf4uFyuek9n1X1u6D6gOqGhobLb7T4LAACwpoAKO0eOHNEXX3yhDh06KDExUS1bttSaNWvM/l27dqmkpERut1uS5Ha7tWPHDpWVlZljVq1aJbvdrvj4+LNePwAACDx+vYx133336cYbb1THjh21b98+TZs2TcHBwbrtttvkcDiUkZGhrKwsRUZGym6365577pHb7VafPn0kSQMHDlR8fLxGjhypvLw8eTweTZ48WZmZmQoNDfXnoQEAgADRqJmd/v3717u8JH1/70vdY+Kn4quvvtJtt92mbt26afjw4YqKitLmzZvVvn17SdKMGTN0ww03KDU1Vf369ZPL5dJrr71mrh8cHKyCggIFBwfL7Xbr9ttv16hRozR9+vTGHBYAALAgm2EYxumuFBQUJI/Ho+joaJ/2srIy/eQnP1F1dXWTFXg2eL1eORwOVVRUNPr+ncTsF5q4KpzLip4Y5e8SAMDyTvXn92ldxvrwww/NX3/88cc+j3fX1NRoxYoV+slP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pRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApfk17OTm5qpXr15q27atoqOjNWTIEO3atctnzLFjx5SZmamoqCiFh4crNTVVpaWlPmNKSkqUkpKi1q1bKzo6WtnZ2Tp+/PjZPBQAABCg/Bp21q9fr8zMTG3evFmrVq1SdXW1Bg4cqKNHj5pjJk6cqGXLlmnx4sVav3699u3bp1tuucXsr6mpUUpKiqqqqrRp0yYtXLhQCxYs0NSpU/1xSAAAIMDYDMMw/F1EnQMHDig6Olrr169Xv379VFFRofbt22vRokUaOnSoJOnTTz9Vjx49VFhYqD59+mj58uW64YYbtG/fPjmdTknS3LlzNWnSJB04cEAhISH/db9er1cOh0MVFRWy2+2Nqj0x+4VGrQdrKnpilL9LAADLO9Wf3wF1z05FRYUkKTIyUpJUVFSk6upqJSUlmWO6d++uuLg4FRYWSpIKCwuVkJBgBh1JSk5Oltfr1c6dOxvcT2Vlpbxer88CAACsKWDCTm1trSZMmKCrr75al112mSTJ4/EoJCREERERPmOdTqc8Ho855sSgU9df19eQ3NxcORwOc4mNjW3iowEAAIEiYMJOZmamPvroI7388svNvq+cnBxVVFSYy969e5t9nwAAwD9a+LsASRo/frwKCgq0YcMGXXjhhWa7y+VSVVWVysvLfWZ3SktL5XK5zDFbt2712V7d01p1Y34oNDRUoaGhTXwUAAAgEPl1ZscwDI0fP16vv/663n77bXXu3NmnPzExUS1bttSaNWvMtl27dqmkpERut1uS5Ha7tWPHDpWVlZljVq1aJbvdrvj4+LNzIAAAIGD5dWYnMzNTixYt0j/+8Q+1bdvWvMfG4XAoLCxMDodDGRkZysrKUmRkpOx2u+655x653W716dNHkjRw4EDFx8dr5MiRysvLk8fj0eTJk5WZmcnsDQAA8G/YmTNnjiTp2muv9WnPz8/X6NGjJUkzZsxQUFCQUlNTVVlZqeTkZD377LPm2ODgYBUUFGjcuHFyu91q06aN0tPTNX369LN1GAAAIIAF1Ht2/IX37KCp8Z4dAGh+5+R7dgAAAJoaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFgaYQcAAFiaX8POhg0bdOONNyomJkY2m01Lly716TcMQ1OnTlWHDh0UFhampKQk7d6922fMwYMHlZaWJrvdroiICGVkZOjIkSNn8SgAAEAg82vYOXr0qK644go988wzDfbn5eVp1qxZmjt3rrZs2aI2bdooOTlZx44dM8ekpaVp586dWrVqlQoKCrRhwwbdeeedZ+sQAABAgGvhz50PHjxYgwcPbrDPMAzNnDlTkydP1k033SRJeuGFF+R0OrV06VKNGDFCn3zyiVasWKFt27apZ8+ekqTZs2fr+uuv1x//+EfFxMQ0uO3KykpVVlaan71ebxMfGQAACBQBe89OcXGxPB6PkpKSzDaHw6HevXursLBQklRYWKiIiAgz6EhSUlKSgoKCtGXLlpNuOzc3Vw6Hw1xiY2Ob70AAAIBfBWzY8Xg8kiSn0+nT7nQ6zT6Px6Po6Gif/hYtWigyMtIc05CcnBxVVFSYy969e5u4egAAECj8ehnLX0JDQxUaGurvMgAAwFkQsDM7LpdLklRaWurTXlpaava5XC6VlZX59B8/flwHDx40xwAAgPNbwIadzp07y+Vyac2aNWab1+vVli1b5Ha7JUlut1vl5eUqKioyx7z99tuqra1V7969z3rNAAAg8Pj1MtaRI0f0+eefm5+Li4u1fft2RUZGKi4uThMmTNAjjzyiSy65RJ07d9aUKVMUExOjIUOGSJJ69OihQYMGaezYsZo7d66qq6s1fvx4jRgx4qRPYgEAgPOLX8POe++9p1/84hfm56ysLElSenq6FixYoPvvv19Hjx7VnXfeqfLycl1zzTVasWKFWrVqZa7z0ksvafz48RowYICCgoKUmpqqWbNmnfVjAQAAgclmGIbh7yL8zev1yuFwqKKiQna7vVHbSMx+oYmrwrms6IlR/i4BACzvVH9+n5dPYwHnC0I4TkQIx/kqYG9QBgAAaAqEHQAAYGlcxgIAnDVcWsWJztalVWZ2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApVkm7DzzzDPq1KmTWrVqpd69e2vr1q3+LgkAAAQAS4SdV155RVlZWZo2bZref/99XXHFFUpOTlZZWZm/SwMAAH5mibDz1FNPaezYsbrjjjsUHx+vuXPnqnXr1po/f76/SwMAAH7Wwt8FnKmqqioVFRUpJyfHbAsKClJSUpIKCwsbXKeyslKVlZXm54qKCkmS1+ttdB01ld81el1Yz5mcS02J8xInCoTzknMSJzrTc7JufcMwfnTcOR92vv76a9XU1MjpdPq0O51Offrppw2uk5ubq4ceeqhee2xsbLPUiPOPY/av/V0CUA/nJQJNU52Thw8flsPhOGn/OR92GiMnJ0dZWVnm59raWh08eFBRUVGy2Wx+rOzc5vV6FRsbq71798put/u7HEAS5yUCD+dk0zEMQ4cPH1ZMTMyPjjvnw067du0UHBys0tJSn/bS0lK5XK4G1wkNDVVoaKhPW0RERHOVeN6x2+38BUbA4bxEoOGcbBo/NqNT55y/QTkkJESJiYlas2aN2VZbW6s1a9bI7Xb7sTIAABAIzvmZHUnKyspSenq6evbsqauuukozZ87U0aNHdccdd/i7NAAA4GeWCDu33nqrDhw4oKlTp8rj8ejKK6/UihUr6t20jOYVGhqqadOm1btECPgT5yUCDefk2Wcz/tvzWgAAAOewc/6eHQAAgB9D2AEAAJZG2AEAAJZG2AEAAJZG2LGo0aNHy2azyWazKSQkRF26dNH06dN1/PhxSVJNTY1mzJihhIQEtWrVShdccIEGDx6sjRs3+mynpqZGjz32mLp3766wsDBFRkaqd+/e+stf/uKzryFDhkiSuc+TLQ8++KC+/PJL2Ww2bd++XUVFRbLZbNq8eXODxzFgwADdcsst9Y7pxGXQoEHN8DuIplb35/fYY4/5tC9dutR8c/m6detOeu54PB5zO3Xn24nq1i0vL9e11177o+fhtddeK0nq1KmT2da6dWslJCT4nNsn+tvf/qbg4GBlZmb+6L4R+JrqXJS+fxvylClTdOmllyosLExRUVHq1auX8vLydOjQoXr7bug8Op3zdebMmaqqqlK7du3q1V/n4YcfltPpVHV1tRYsWNDgNlu1anWmv43nFEs8eo6GDRo0SPn5+aqsrNSbb76pzMxMtWzZUg888IBGjBih1atX64knntCAAQPk9Xr1zDPP6Nprr9XixYvNHyYPPfSQ5s2bp6efflo9e/aU1+vVe++91+BfYknav3+/+etXXnlFU6dO1a5du8y28PBwff311+bnxMREXXHFFZo/f7769Onjs60vv/xSa9eu1bJly+od04l4fPPc0apVKz3++OO66667dMEFF5x03K5du+q9WTY6OvqU9/Paa6+pqqpKkrR3715dddVVWr16tS699FJJ37+MtM706dM1duxYffvtt1q8eLHGjh2rn/zkJxo8eLDPNp9//nndf//9mjdvnp588snz7oeF1TTFuXjw4EFdc8018nq9evjhh5WYmCiHw6Fdu3YpPz9fixYtqheOGzqPTud8rft8++23Kz8/Xw888IBPn2EYWrBggUaNGqWWLVtK+v5NzSf+OyzpvPtqJMKOhYWGhppfmTFu3Di9/vrreuONN3TRRRdpyZIleuONN3TjjTea45977jl98803+tWvfqXrrrtObdq00RtvvKG7775bw4YNM8ddccUVJ93niV/R4XA4ZLPZ6n1tx4lhR5IyMjI0efJkzZw5U61btzbbFyxYoA4dOvjM3Jx4TDj3JCUl6fPPP1dubq7y8vJOOi46OvqMvsIlMjLS/PWxY8ckSVFRUQ2eO23btjXbJ02apLy8PK1atcon7BQXF2vTpk36+9//rrVr1+q1117TL3/5y0bXB/9rinPxd7/7nUpKSvTZZ5/5fDdTx44dNXDgwHrfxH2y8+h0ztc6GRkZ+tOf/qR3331X11xzjdm+fv16/fvf/1ZGRobZ1tC/w+cbLmOdR8LCwlRVVaVFixapa9euPkGnzr333qtvvvlGq1atkvR9eHn77bd14MCBZqsrLS1NlZWVWrJkidlmGIYWLlyo0aNHKzg4uNn2jbMrODhYjz76qGbPnq2vvvrK3+X4qK2t1d///ncdOnSo3v+k8/PzlZKSIofDodtvv13PP/+8n6pEUznTc7G2tlavvPKKbr/99pN+CeUPZ0+a8jxKSEhQr169NH/+/Hr76Nu3r7p3797obVsRYec8YBiGVq9erbfeekv9+/fXZ599ph49ejQ4tq79s88+kyQ99dRTOnDggFwuly6//HL9+te/1vLly5u0vsjISN18880+f2nXrl2rL7/8st5XfhQUFCg8PNxnefTRR5u0HjSvm2++WVdeeaWmTZt20jEXXnihz59x3XR+c5g0aZLCw8MVGhqqoUOH6oILLtCvfvUrs7+2tlYLFizQ7bffLkkaMWKE3n33XRUXFzdbTTg7zuRcPHDggMrLy9WtWzef8YmJiebY2267zWxvjvMoIyNDixcv1pEjRyRJhw8f1pIlSzRmzBifcRUVFfX+3fzhZVqr4zKWhdUFg+rqatXW1uqXv/ylHnzwQRUUFNSbXj2Z+Ph4ffTRRyoqKtLGjRu1YcMG3XjjjRo9evRJb+RsjDFjxig5OVlffPGFLr74Ys2fP18///nP1aVLF59xv/jFLzRnzhyfthOngHFuePzxx9W/f3/dd999Dfa/8847atu2rfm57t6D5pCdna3Ro0dr//79ys7O1t133+1z3q1atUpHjx7V9ddfL0lq166drrvuOs2fP18PP/xws9WFs6Opz8XXX39dVVVVmjRpkr777juzvTnOo9tuu00TJ07Uq6++qjFjxuiVV15RUFCQbr31Vp9xbdu21fvvv+/TFhYW1qh9nqsIOxZWFwxCQkIUExOjFi2+/+Pu2rWrPvnkkwbXqWvv2rWr2RYUFKRevXqpV69emjBhgl588UWNHDlSv//979W5c+cmqXXAgAGKi4vTggULlJ2drddee03z5s2rN65Nmzb1AhDOPf369VNycrJycnI0evToev2dO3c+6X0Sdrtde/bsqddeXl6u4OBgtWnT5rRqadeunbp06aIuXbpo8eLFSkhIUM+ePRUfHy/p+xtKDx486PPDoba2Vh9++KEeeughBQUxQX4ua+y52L59e0VERNS78TcuLk7S9wHjxKfzmuM8stvtGjp0qPLz8zVmzBjl5+dr+PDhCg8P9xkXFBR03v+7yd9SC6sLBnFxcWbQkb6fPt29e7fPU051nnzySUVFRem666476XbrfggcPXq0yWoNCgrSHXfcoYULF2rRokUKCQnR0KFDm2z7CDyPPfaYli1bpsLCwtNar1u3btq5c6cqKyt92t9//3117tz5jGaBYmNjdeuttyonJ0eS9M033+gf//iHXn75ZW3fvt1c/vWvf+nQoUNauXJlo/eFwNGYczEoKEjDhw/Xiy++qH379v3o2OY8jzIyMvTuu++qoKBAmzZt8rkxGf+HmZ3z0IgRI7R48WKlp6fXe/T8jTfe0OLFi83/HQ8dOlRXX321+vbtK5fLpeLiYuXk5Khr165NfgPcHXfcoenTp+t3v/udbrvttganWSsrK33ecSFJLVq0ULt27Zq0FjS/hIQEpaWladasWfX6ysrKzKdS6kRFRally5ZKS0vT9OnTNWrUKN1///1yOBzasGGDZs6c+aNP1Zyq3/72t7rsssv03nvv6d1331VUVJSGDx9e72bT66+/Xs8//7zP04I7duzwueRhs9l+9OlFBIbGnouPPvqo1q1bp6uuukrTp09Xz5491aZNG3344YcqLCzUZZddJkn661//elrn0eno16+funTpolGjRql79+7q27dvvTGGYdT7d1P6/kmz82VmkrBzHrLZbHr11Vc1c+ZMzZgxQ3fffbdatWolt9utdevW6eqrrzbHJicn629/+5tyc3NVUVEhl8ul/v3768EHH/SZLWoKcXFxSkpK0sqVK+vdYFdnxYoV6tChg09bt27d9OmnnzZpLTg7pk+frldeeaVe+w9v+pSkwsJC9enTRxEREXrnnXf0wAMP6H//939VUVGhLl266KmnnmqS/9XGx8dr4MCBmjp1qr766ivdfPPNDb6TJDU1VSNHjvR5lUK/fv18xgQHB5sv8kRga8y5GBUVpa1bt+rxxx/XE088oeLiYgUFBemSSy7RrbfeqgkTJkiS5s+ff0rnUWP+02az2TRmzBj97ne/M2ckf8jr9db7d1P6/r1o58sj6TbjVO9UBQAAOAedH/NXAADgvEXYAQAAlkbYAQAAlkbYAQAAlkbYAQAAlkbYAQAAlkbYAQAAlkbYAQAAlkbYAWA569atk81m8/kiRgDnL8IOgGZz4MABjRs3TnFxcQoNDZXL5VJycrI2btzYZPu49tprzdfy1+nbt6/2798vh8PRZPtprNGjR2vIkCH+LgM4r/HdWACaTWpqqqqqqrRw4UJddNFFKi0t1Zo1a/TNN980635DQkLOm+/8AXAKDABoBocOHTIkGevWrfvRMRkZGUa7du2Mtm3bGr/4xS+M7du3m/3Tpk0zrrjiCuOFF14wOnbsaNjtduPWW281vF6vYRiGkZ6ebkjyWYqLi421a9cakoxDhw4ZhmEY+fn5hsPhMJYtW2Z07drVCAsLM1JTU42jR48aCxYsMDp27GhEREQY99xzj3H8+HFz/8eOHTPuvfdeIyYmxmjdurVx1VVXGWvXrjX767a7YsUKo3v37kabNm2M5ORkY9++fWb9P6zvxPUBnB1cxgLQLMLDwxUeHq6lS5eqsrKywTHDhg1TWVmZli9frqKiIv3sZz/TgAEDdPDgQXPMF198oaVLl6qgoEAFBQVav369HnvsMUnSn/70J7ndbo0dO1b79+/X/v37FRsb2+C+vv32W82aNUsvv/yyVqxYoXXr1unmm2/Wm2++qTfffFN//etfNW/ePC1ZssRcZ/z48SosLNTLL7+sDz/8UMOGDdOgQYO0e/dun+3+8Y9/1F//+ldt2LBBJSUluu+++yRJ9913n4YPH65BgwaZ9fXt2/eMf28BnCZ/py0A1rVkyRLjggsuMFq1amX07dvXyMnJMT744APDMAzjnXfeMex2u3Hs2DGfdS6++GJj3rx5hmF8PzPSunVrcybHMAwjOzvb6N27t/n55z//ufHb3/7WZxsNzexIMj7//HNzzF133WW0bt3aOHz4sNmWnJxs3HXXXYZhGMaePXuM4OBg4z//+Y/PtgcMGGDk5OScdLvPPPOM4XQ6zc/p6enGTTfddEq/XwCaB/fsAGg2qampSklJ0TvvvKPNmzdr+fLlysvL01/+8hcdPXpUR44cUVRUlM863333nb744gvzc6dOndS2bVvzc4cOHVRWVnbatbRu3VoXX3yx+dnpdKpTp04KDw/3aavb9o4dO1RTU6OuXbv6bKeystKn5h9ut7H1AWg+hB0AzapVq1a67rrrdN1112nKlCn61a9+pWnTpunuu+9Whw4dtG7dunrrREREmL9u2bKlT5/NZlNtbe1p19HQdn5s20eOHFFwcLCKiooUHBzsM+7EgNTQNgzDOO36ADQfwg6Asyo+Pl5Lly7Vz372M3k8HrVo0UKdOnVq9PZCQkJUU1PTdAX+fz/96U9VU1OjsrIy/c///E+jt9Nc9QE4ddygDKBZfPPNN+rfv79efPFFffjhhyouLtbixYuVl5enm266SUlJSXK73RoyZIhWrlypL7/8Ups2bdLvf/97vffee6e8n06dOmnLli368ssv9fXXXzdq1qchXbt2VVpamkaNGqXXXntNxcXF2rp1q3Jzc/XPf/7ztOr78MMPtWvXLn399deqrq5ukvoAnDrCDoBmER4ert69e2vGjBnq16+fLrvsMk2ZMkVjx47V008/LZvNpjfffFP9+vXTHXfcoa5du2rEiBHas2ePnE7nKe/nvvvuU3BwsOLj49W+fXuVlJQ02THk5+dr1KhRuvfee9WtWzcNGTJE27ZtU1xc3ClvY+zYserWrZt69uyp9u3bN+kLFQGcGpvBxWUAAGBhzOwAAABLI+wAAABLI+wAAABLI+wAAABLI+wAAABLI+wAAABLI+wAAABLI+wAAABLI+wAAABLI+wAAABLI+wAAABL+3/7UgMXeLsU4gAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##### observation\n",
        "* Number of positive reviews is very high as compared to negative or neutral. So the data is imbalanced. And we cannot use accuracy as an evaluation matrix."
      ],
      "metadata": {
        "id": "7FlMsSqvBm8O"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df['Sentiment'].value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 209
        },
        "id": "Y9yu6p2Qe3u3",
        "outputId": "57107d98-a3bf-43bb-dd5c-cbdb35b613f2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Sentiment\n",
              "POSITIVE    850\n",
              "NEUTRAL      81\n",
              "NEGATIVE     74\n",
              "Name: count, dtype: int64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>count</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Sentiment</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>POSITIVE</th>\n",
              "      <td>850</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>NEUTRAL</th>\n",
              "      <td>81</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>NEGATIVE</th>\n",
              "      <td>74</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> int64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df['Sentiment'].value_counts(normalize=True)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 209
        },
        "id": "k6_Da1Hie9o5",
        "outputId": "af3f3e96-4bae-4000-884d-7bce4f54c2b1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Sentiment\n",
              "POSITIVE    0.845771\n",
              "NEUTRAL     0.080597\n",
              "NEGATIVE    0.073632\n",
              "Name: proportion, dtype: float64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>proportion</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Sentiment</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>POSITIVE</th>\n",
              "      <td>0.845771</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>NEUTRAL</th>\n",
              "      <td>0.080597</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>NEGATIVE</th>\n",
              "      <td>0.073632</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> float64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Majority of the reviews are positive (\\~85%), followed by neutral reviews (8%), and then the positive reviews (\\~7%)"
      ],
      "metadata": {
        "id": "RYeG4DIft2X8"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Recommended Metrics for this Case:\n",
        "| Metric                               | Why It's Important                                                                  |\n",
        "| ------------------------------------ | ----------------------------------------------------------------------------------- |\n",
        "| **Macro F1-Score**                   | Gives equal importance to all 3 classes regardless of imbalance.                    |\n",
        "| **Per-class Precision & Recall**     | Helps you understand how well the model detects **Neutral** and **Negative** cases. |\n",
        "| **Confusion Matrix**                 | Shows what types of mistakes your model is making.                                  |\n",
        "| *(Optional)* **ROC-AUC (per class)** | Can be helpful if you're using probabilistic outputs.                               |\n"
      ],
      "metadata": {
        "id": "b8VPWUo5vpUS"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### Macro F1 vs Micro F1\n",
        "| Type         | Use When                                              | What It Does                           |\n",
        "| ------------ | ----------------------------------------------------- | -------------------------------------- |\n",
        "| **Macro F1** | Treat all classes equally (class-balanced evaluation) | Averages F1 across all classes         |\n",
        "| **Micro F1** | Use when class sizes vary (class-imbalanced)          | Calculates global counts of TP, FP, FN |\n"
      ],
      "metadata": {
        "id": "FKDqsPFbvxM_"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Test Preprocessing**"
      ],
      "metadata": {
        "id": "wmBfpI30FBRb"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Removing special characters from the text**"
      ],
      "metadata": {
        "id": "WHwKGLoNF8_9"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# defining a func to remove special chars\n",
        "def remove_special_characters(text):\n",
        "  #defining the regex pattern to match the non-alphanumeric chars\n",
        "  pattern = '[^A-Za-z0-9]+'\n",
        "\n",
        "  #finding the specified
