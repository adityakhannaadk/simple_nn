{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "kernelspec": {
      "display_name": "Python (tensorflow)",
      "language": "python",
      "name": "tensorflow"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.6.7"
    },
    "colab": {
      "name": "neuraname_d1.ipynb",
      "provenance": [],
      "include_colab_link": true
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
        "<a href=\"https://colab.research.google.com/github/adityakhannaadk/simple_nn/blob/master/neuraname_d1.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "j73D4q3ioA9P",
        "colab_type": "code",
        "outputId": "b96beada-b007-45c2-99f4-e88c04b27f1d",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 326
        }
      },
      "source": [
        "!pip install py-lorem\n",
        "from tensorflow.keras.layers import LSTM, Dense, Input, concatenate, Reshape, Dropout\n",
        "from tensorflow.keras.models import Model, load_model\n",
        "import numpy as np\n",
        "%tensorflow_version 1.4"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Collecting py-lorem\n",
            "  Downloading https://files.pythonhosted.org/packages/15/c2/13bc93a872db382b8f6bca3ed132ded7520dd37171a4ed34275bc09aae59/py-lorem-1.2.tar.gz\n",
            "Building wheels for collected packages: py-lorem\n",
            "  Building wheel for py-lorem (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for py-lorem: filename=py_lorem-1.2-cp36-none-any.whl size=2744 sha256=61d608a42611d78b4993bf6ffb4f6cc8a0d2c5ba9d71c3b12b6fa16708e92ea2\n",
            "  Stored in directory: /root/.cache/pip/wheels/0d/96/08/333da2ca0964fae44e4377acfeb049d2f575aa22d63f5207c4\n",
            "Successfully built py-lorem\n",
            "Installing collected packages: py-lorem\n",
            "Successfully installed py-lorem-1.2\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "text/html": [
              "<p style=\"color: red;\">\n",
              "The default version of TensorFlow in Colab will soon switch to TensorFlow 2.x.<br>\n",
              "We recommend you <a href=\"https://www.tensorflow.org/guide/migrate\" target=\"_blank\">upgrade</a> now \n",
              "or ensure your notebook will continue to use TensorFlow 1.x via the <code>%tensorflow_version 1.x</code> magic:\n",
              "<a href=\"https://colab.research.google.com/notebooks/tensorflow_version.ipynb\" target=\"_blank\">more info</a>.</p>\n"
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "stream",
          "text": [
            "`%tensorflow_version` only switches the major version: `1.x` or `2.x`.\n",
            "You set: `1.4`. This will be interpreted as: `1.x`.\n",
            "\n",
            "\n",
            "TensorFlow is already loaded. Please restart the runtime to change versions.\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VMFjN0fhoA9T",
        "colab_type": "code",
        "outputId": "048a4f42-bfe5-452d-9992-8842f57d21c5",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 130
        }
      },
      "source": [
        "def process_names(names,*,unwanted=['(', ')', '-', '.', '/']):\n",
        "    names = [name.lower() for name in names]\n",
        "    print(\"Total names:\",len(names))\n",
        "    chars = sorted(list(set(''.join(names))))\n",
        "\n",
        "    def has_unwanted(word):\n",
        "        for char in word:\n",
        "            if char in unwanted:\n",
        "                return True\n",
        "        return False\n",
        "    names = [name for name in names if not has_unwanted(name)]\n",
        "    print(\"Amount of names after removing those with unwanted characters\\n:\",len(names))\n",
        "    chars = [char for char in chars if char not in unwanted]\n",
        "    print(\"Using the following characters:\\n\",chars)\n",
        "\n",
        "    maxlen = max([len(name) for name in names])\n",
        "    minlen = min([len(name) for name in names])\n",
        "\n",
        "  \n",
        "    endchars = '!£$%^&*()-_=+/?.>,<;:@[{}]#~'\n",
        "    endchar = [ch for ch in endchars if ch not in chars][0]\n",
        "\n",
        "\n",
        "    assert(endchar not in chars)\n",
        "    chars += endchar\n",
        "    \n",
        "    return names,chars\n",
        "\n",
        "names = '''\n",
        "Stepwells\n",
        "A millennium ago, stepwells were fundamental to life in the driest parts of India. Richard Cox travelled to north- western India to document these spectacular monuments from a bygone era\n",
        "During the sixth and seventh centuries, the inhabitants of the modern-day states of Gujarat and Rajasthan in north-western India developed a method of gaining access to clean, fresh groundwater during the dry season for drinking, bathing, watering animals and irrigation. However, the significance of this invention - the stepwell - goes beyond its utilitarian application.\n",
        "Unique to this region, stepwells are often architecturally complex and vary widely in size and shape. During their heyday, they were places of gathering, of leisure and relaxation and of worship for villagers of all but the lowest classes. Most stepwells are found dotted round the desert areas of Gujarat (where they are called vav) and Rajasthan (where they are called baori}, while a few also survive in Delhi. Some were located in or near villages as public spaces for the community; others were positioned beside roads as resting places for travellers.\n",
        "As their name suggests, stepwells comprise a series of stone steps descending from ground level to the water source (normally an underground aquifer) as it recedes following the rains. When the water level was high, the user needed only to descend a few steps to reach it; when it was low, several levels would have to be negotiated.\n",
        "Some wells are vast, open craters with hundreds of steps paving each sloping side, often in tiers. Others are more elaborate, with long stepped passages leading to the water via several storeys. Built from stone and supported by pillars, they also included pavilions that sheltered visitors from the relentless heat. But perhaps the most impressive features are the intricate decorative sculptures that embellish many stepwells, showing activities from fighting and dancing to everyday acts such as women combing their hair or churning butter.\n",
        "Down the centuries, thousands of wells were constructed throughout north­ western India, but the majority have now fallen into disuse; many are derelict and dry, as groundwater has been diverted for industrial use and the wells no longer reach the water table. Their condition hasn't been helped by recent dry spells: southern Rajasthan suffered an eight-year drought between 1996 and 2004.\n",
        "However, some important sites in Gujarat have recently undergone major restoration, and the state government announced in June last year that it plans to restore the stepwells throughout the state.\n",
        "In Patan, the state's ancient capital, the stepwell of Rani Ki Vav (Queen's Stepwell) is perhaps the finest current example. It was built by Queen Udayamati during the late 11th century, but became silted up following a flood during the 13th century. But the Archaeological Survey of India began restoring it in the 1960s, and today it is in pristine condition. At 65 metres long, 20 metres wide and 27 metres deep, Rani Ki Vav features 500 sculptures carved into niches throughout the monument.  Incredibly, in January 2001, this ancient structure survived an earthquake that measured 7.6 on the Richter scale.\n",
        "Another example is the Surya Kund in Modhera, northern Gujarat, next to the Sun Temple, built by King Bhima I in 1026 to honour the sun god Surya. It actually resembles a tank (kund means reservoir or pond) rather than a well, but displays the hallmarks of stepwell architecture, including four sides of steps that descend to the bottom in a stunning geometrical formation. The terraces house 108 small, intricately carved shrines between the sets of steps.\n",
        "Rajasthan also has a wealth of wells. The ancient city of Bundi, 200 kilometres south of Jaipur, is renowned for its architecture, including its stepwells. One of the larger examples is Raniji Ki Baori, which was built by the queen of the region, Nathavatji, in 1699. At 46 metres deep, 20 metres wide and 40 metres long, the intricately carved monument is one of 21 baoris commissioned in the Bundi area by Nathavatji.\n",
        "In the old ruined town of Abhaneri, about 95 kilometres east of Jaipur, is Chand  Baori, one of India's oldest and deepest wells; aesthetically it's perhaps one of the most dramatic. Built in around  850 AD next to the temple of Harshat Mata, the  baori comprises hundreds of zigzagging steps that run along three  of its sides, steeply   descending   11storeys,   resulting in a striking pattern when seen from  afar. On the fourth side, verandas which are supported by ornate pillars overlook the steps.\n",
        "Still in public use is Neemrana Ki Baori, located just off the Jaipur-Delhi highway. Constructed in around 1700, it is nine storeys deep, with the last two being underwater. At ground level, there are 86 colonnaded openings from where the visitor descends 170 steps to the deepest water source.\n",
        "Today, following years of neglect, many of these monuments to medieval engineering have been saved by the Archaeological Survey of India, which has recognised the importance of preserving them as part of the country’s rich history. Tourists flock to wells in far-flu ng corners of north­ western India to gaze in wonder at these architectural marvels from hundreds of years ago, which serve as a reminder of both the ingenuity and artistry of ancient civilisations and of the value of water to human existence.\n",
        "'''.split(\" \")\n",
        "names,chars = process_names(names)"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Total names: 876\n",
            "Amount of names after removing those with unwanted characters\n",
            ": 822\n",
            "Using the following characters:\n",
            " ['\\n', \"'\", ',', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '}', '\\xad', '’']\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3lC4OP6LoA9W",
        "colab_type": "code",
        "outputId": "ba0d12d5-7ea1-4c9d-ad63-f8010d7a6391",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 36
        }
      },
      "source": [
        "def make_sequences(names,seqlen):\n",
        "    sequences, lengths, nextchars = [],[],[] # To have the model learn a more macro understanding, \n",
        "                                             # it also takes the word's length so far as input\n",
        "    for name in names:\n",
        "        if len(name) <= seqlen:\n",
        "            sequences.append(name + chars[-1]*(seqlen - len(name)))\n",
        "            nextchars.append(chars[-1])\n",
        "            lengths.append(len(name))\n",
        "        else:\n",
        "            for i in range(0,len(name)-seqlen+1):\n",
        "                sequences.append(name[i:i+seqlen])\n",
        "                if i+seqlen < len(name):\n",
        "                    nextchars.append(name[i+seqlen])\n",
        "                else:\n",
        "                    nextchars.append(chars[-1])\n",
        "                lengths.append(i+seqlen)\n",
        "\n",
        "    print(len(sequences),\"sequences of length\",seqlen,\"made\")\n",
        "    \n",
        "    return sequences,lengths,nextchars\n",
        "\n",
        "seqlen = 4\n",
        "sequences,lengths,nextchars = make_sequences(names,seqlen)"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "2114 sequences of length 4 made\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "h8sYpigroA9Y",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def make_onehots(*,sequences,lengths,nextchars,chars):\n",
        "    x = np.zeros(shape=(len(sequences),len(sequences[0]),len(chars)), dtype='float32') # sequences\n",
        "    x2 = np.zeros(shape=(len(lengths),max(lengths))) # lengths\n",
        "\n",
        "    for i, seq in enumerate(sequences):\n",
        "        for j, char in enumerate(seq):\n",
        "            x[i,j,chars.index(char)] = 1.\n",
        "\n",
        "    for i, l in enumerate(lengths):\n",
        "        x2[i,l-1] = 1.\n",
        "\n",
        "    y = np.zeros(shape=(len(nextchars),len(chars)))\n",
        "    for i, char in enumerate(nextchars):\n",
        "        y[i,chars.index(char)] = 1.\n",
        "    \n",
        "    return x,x2,y\n",
        "\n",
        "x,x2,y = make_onehots(sequences=sequences,\n",
        "                     lengths=lengths,\n",
        "                     nextchars=nextchars,\n",
        "                     chars=chars)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ucCIosPioA9a",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def get_dictchars(names,seqlen):\n",
        "    dictchars = [{} for _ in range(seqlen)]\n",
        "\n",
        "    for name in names:\n",
        "        if len(name) < seqlen:\n",
        "            continue\n",
        "        dictchars[0][name[0]] = dictchars[0].get(name[0],0) + 1\n",
        "        for i in range(1,seqlen):\n",
        "            if dictchars[i].get(name[i-1],0) == 0:\n",
        "                dictchars[i][name[i-1]] = {name[i]: 1}\n",
        "            elif dictchars[i][name[i-1]].get(name[i],0) == 0:\n",
        "                dictchars[i][name[i-1]][name[i]] = 1\n",
        "            else:\n",
        "                dictchars[i][name[i-1]][name[i]] += 1\n",
        "    return dictchars\n",
        "                \n",
        "dictchars = get_dictchars(names,seqlen)\n",
        "                \n",
        "'''\n",
        "What is dictchars?\n",
        "Basically, stores how often a letter occurs after another letter at a specific spot in a name\n",
        "\n",
        "dictchars[0] just stores how often each letter is first, {a: 3, b:4, etc}\n",
        "\n",
        "dictchars[1+] store which letters (and how often) come after a certain letter.\n",
        "For example, if dictchars[1]['a'] = {b:4,c:1}, that means that if 'a' was first, \n",
        "b followed 4 times, while c followed only once.\n",
        "\n",
        "This is used in the method below to generate plausible-sounding starting sequences.\n",
        "'''\n",
        "    \n",
        "\n",
        "def generate_start_seq(dictchars):\n",
        "    res = \"\" # The starting sequence will be stored here\n",
        "    p = sum([n for n in dictchars[0].values()]) # total amount of letter occurences\n",
        "    r = np.random.randint(0,p) # random number used to pick the next character\n",
        "    tot = 0\n",
        "    for key, item in dictchars[0].items():\n",
        "        if r >= tot and r < tot + item:\n",
        "            res += key\n",
        "            break\n",
        "        else:\n",
        "            tot += item\n",
        "\n",
        "    for i in range(1,len(dictchars)):\n",
        "        ch = res[-1]\n",
        "        if dictchars[i].get(ch,0) == 0:\n",
        "            l = list(dictchars[i].keys())\n",
        "            ch = l[np.random.randint(0,len(l))]\n",
        "        p = sum([n for n in dictchars[i][ch].values()])\n",
        "        r = np.random.randint(0,p)\n",
        "        tot = 0\n",
        "        for key, item in dictchars[i][ch].items():\n",
        "            if r >= tot and r < tot + item:\n",
        "                res += key\n",
        "                break\n",
        "            else:\n",
        "                tot += item\n",
        "    return res\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1K6lnYh7oA9c",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def sample(preds,temperature=0.4):\n",
        "    preds = np.asarray(preds).astype('float64')\n",
        "    if temperature == 0:\n",
        "        # Avoiding a division by 0 error\n",
        "        return np.argmax(preds)\n",
        "    preds = np.log(preds) / temperature\n",
        "    exp_preds = np.exp(preds)\n",
        "    preds = exp_preds / np.sum(exp_preds)\n",
        "    probas = np.random.multinomial(1,preds,1)\n",
        "    return np.argmax(probas)\n",
        "\n",
        "def generate_name(model,start,*,chars=chars,temperature=0.4):\n",
        "    maxlength = model.layers[3].input.shape[1]\n",
        "    seqlen = int(model.layers[0].input.shape[1])\n",
        "    result = start\n",
        "    \n",
        "    sequence_input = np.zeros(shape=(1,seqlen,len(chars)))\n",
        "    for i, char in enumerate(start):\n",
        "        sequence_input[0,i,chars.index(char)] = 1.\n",
        "    \n",
        "    length_input = np.zeros(shape=(1,maxlength))\n",
        "    length_input[0,len(result)-1] = 1.\n",
        "    \n",
        "    prediction = model.predict(x=[sequence_input,length_input])[0]\n",
        "    char_index = sample(prediction,temperature)\n",
        "    while char_index < len(chars)-1 and len(result) < maxlength:\n",
        "        result += chars[char_index]\n",
        "        \n",
        "        sequence_input = np.zeros(shape=(1,seqlen,len(chars)))\n",
        "        for i, char in enumerate(result[(-seqlen):]):\n",
        "            sequence_input[0,i,chars.index(char)] = 1.\n",
        "        \n",
        "        length_input[0,len(result)-2] = 0.\n",
        "        length_input[0,len(result)-1] = 1.\n",
        "        \n",
        "        prediction = model.predict(x=[sequence_input,length_input])[0]\n",
        "        char_index = sample(prediction,temperature)\n",
        "    \n",
        "    return result.title()\n",
        "\n",
        "def generate_random_name(model,*,chars=chars,dictchars=dictchars,temperature=0.4):\n",
        "    start = generate_start_seq(dictchars)\n",
        "    return generate_name(model,start,chars=chars,temperature=temperature)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Fj9W7s-coA9e",
        "colab_type": "code",
        "outputId": "09cd0603-9c9b-4eb5-90b8-69f7f201b9ce",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 93
        }
      },
      "source": [
        "def make_model(x,x2,chars):\n",
        "    inp1 = Input(shape=x.shape[1:]) # sequence input\n",
        "    inp2 = Input(shape=x2.shape[1:]) # length input\n",
        "    lstm = LSTM(len(chars),activation='relu',dropout=0.3)(inp1)\n",
        "    lstm2 = LSTM(len(chars),dropout=0.3,go_backwards=True)(inp1)\n",
        "    concat = concatenate([lstm,lstm2,inp2])\n",
        "    dense = Dense(len(chars),activation='softmax')(concat)\n",
        "\n",
        "    model = Model([inp1,inp2],dense)\n",
        "    model.compile(optimizer='adam',loss='binary_crossentropy')\n",
        "    return model\n",
        "\n",
        "model = make_model(x,x2,chars)"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.\n",
            "Instructions for updating:\n",
            "If using Keras pass *_constraint arguments to layers.\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LiURUiKEoA9h",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "\n",
        "def use_model(model,*,x=x,x2=x2,y=y,chars=chars,dictchars=dictchars,total_epochs=180,print_every=60,temperature=0.4,verbose=True):\n",
        "    for i in range(total_epochs//print_every):\n",
        "        history = model.fit([x,x2],y,\n",
        "                            epochs=print_every,\n",
        "                            batch_size=64,\n",
        "                            validation_split=0.05,\n",
        "                            verbose=0)\n",
        "        list_of_names = []\n",
        "        if verbose:\n",
        "            for _ in range(30):\n",
        "                list_of_names.append(generate_random_name(model,chars=chars,dictchars=dictchars,temperature=temperature)) \n",
        "    if not verbose:\n",
        "  \n",
        "        for _ in range(20):\n",
        "            print(generate_random_name(model,chars=chars,dictchars=dictchars,temperature=0.4))\n",
        "    return(list_of_names)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "odW3YGLroA9j",
        "colab_type": "code",
        "outputId": "47feda96-9206-46cf-dff3-c3e774199d31",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 112
        }
      },
      "source": [
        "m9_wot = use_model(model)\n",
        "print(m9_wot)"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.\n",
            "Instructions for updating:\n",
            "Use tf.where in 2.0, which has the same broadcast rule as np.where\n",
            "['Desorn', 'Sthaps', '1996', 'Mouned', 'Rean', 'Caoundwater', 'Stepwells', 'Suffernd', 'Pajorites', '11Scending', 'Sitant', 'Oloning,', 'Thrse', 'Intricately', 'Wero', 'Dicreds', 'Beaces', 'Cots', 'Begent', 'Vate', 'Dond', 'Whach', 'Kisuments', 'Leach', 'Weades', 'Wenored', 'Ruchern', 'Momites', 'Shepped', 'Songhere']\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Sl8IZb5HojYJ",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "SIMILAR = [\n",
        "  ['a', 'e', 'i', 'o'],\n",
        "  ['c', 'k'],\n",
        "  ['j', 'g', 'ch'],\n",
        "  ['s', 'j', 'z'],\n",
        "  ['t', 'd'],\n",
        "  ['n', 'ng']\n",
        "]\n",
        "\n",
        "TOKS = {\n",
        "  'ch',\n",
        "  'ng',\n",
        "}\n",
        "\n",
        "RATE_LEN = 2\n",
        "\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "cvP7uLH2oq8N",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def tokenize(string):\n",
        "  ind = 0\n",
        "  while ind < len(string):\n",
        "    for tok in TOKS:\n",
        "      if string.startswith(tok, ind):\n",
        "        ind += len(tok)\n",
        "        yield tok\n",
        "        break\n",
        "    else:\n",
        "      yield string[ind]\n",
        "      ind += 1\n",
        "\n",
        "def similar(tok1, tok2):\n",
        "  for s in SIMILAR:\n",
        "    if tok1 in s and tok2 in s:\n",
        "      return True\n",
        "  return False\n",
        "\n",
        "def rate_similarity(toks1, toks2):\n",
        "  res = 0\n",
        "  for tok1, tok2 in zip(toks1, toks2):\n",
        "    if similar(tok1, tok2):\n",
        "      res += 1\n",
        "  return res\n",
        "\n",
        "\n",
        "\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SSUpalWrqT8p",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def combine(toks1, toks2):\n",
        "  flag = True\n",
        "  for tok1, tok2 in zip(toks1, toks2):\n",
        "    if flag:\n",
        "      yield tok1\n",
        "    else:\n",
        "      yield tok2\n",
        "    flag = not flag\n",
        "\n",
        "def get_names(pre, suf):\n",
        "  for p_ind in range(len(pre)):\n",
        "    for s_ind in range(len(suf)):\n",
        "      mid1, mid2 = pre[p_ind:p_ind + RATE_LEN], suf[s_ind - RATE_LEN:s_ind]\n",
        "      yield rate_similarity(mid1, mid2), pre[:p_ind] + list(combine(mid1, mid2)) + suf[s_ind:]\n",
        "\n",
        "def get_best_name(pre, suf):\n",
        "  return max(get_names(pre, suf))\n",
        "\n",
        "def ship(name1, name2):\n",
        "  name1, name2 = list(tokenize(name1)), list(tokenize(name2))\n",
        "  score1, ship1 = get_best_name(name1, name2)\n",
        "  score2, ship2 = get_best_name(name2, name1)\n",
        "  if score1 < score2:\n",
        "    return ''.join(ship2)\n",
        "  else:\n",
        "    return ''.join(ship1)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Mslg6lZLqVEY",
        "colab_type": "code",
        "outputId": "67e902d2-cc54-469d-abc7-0a7c2d9ed88c",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 577
        }
      },
      "source": [
        "for wot in m9_wot:\n",
        "  print(ship(wot,\"amazon\"))\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Dezon\n",
            "Sthon\n",
            "mazon\n",
            "Mozon\n",
            "Ron\n",
            "Con\n",
            "Ston\n",
            "Suffon\n",
            "Pazon\n",
            "amazonding\n",
            "amazont\n",
            "amazong,\n",
            "Thrson\n",
            "Intron\n",
            "Won\n",
            "Don\n",
            "Beacezon\n",
            "Cozon\n",
            "amazont\n",
            "Von\n",
            "amazond\n",
            "Whon\n",
            "Kizon\n",
            "Lon\n",
            "Weadezon\n",
            "amazonored\n",
            "Ruchon\n",
            "Momitezon\n",
            "Shon\n",
            "amazonghere\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "snVu-iJWxKoz",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "oWW2llx3x6pV",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}