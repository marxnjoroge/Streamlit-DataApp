import streamlit as st
from bs4 import  BeautifulSoup as bs
import pandas as pd
import requests as rq
import plotly.graph_objects as go
import json
import streamlit.components.v1 as components
import random
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from datetime import datetime,timedelta

st.set_page_config(layout="wide", page_title="Marxian Lab")

padding = 2
st.markdown(f""" <style>    
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

hide_menu_style = """ <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden; }
        </style> """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# @st.cache
def load_data():
    url = "https://finviz.com/crypto_performance.ashx"
    html = pd.read_html(url, header=0)
    dfcrypto = html[0]
    st.dataframe(dfcrypto)
    return dfcrypto


# Sorting Algorithms.

def bubbleSort(arr):

    yield arr
    for i in range(len(arr) - 1, 0, -1):
        for j in range(i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
            yield (*arr,)


def mergeSort(arr):

    if len(arr) > 1:
        mid = len(arr) // 2
        lefthalf = arr[:mid]
        righthalf = arr[mid:]

        yield from mergeSort(lefthalf)
        yield from mergeSort(righthalf)

        # These are indexes: i for lefthalf, j for righthalf, k for nlist.
        i = j = k = 0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                arr[k] = lefthalf[i]
                i = i+1
            else:
                arr[k] = righthalf[j]
                j = j+1
            k = k+1
            yield (*arr,)
        while i < len(lefthalf):
            arr[k] = lefthalf[i]
            i = i+1
            k = k+1
            yield (*arr,)
        while j < len(righthalf):
            arr[k] = righthalf[j]
            j = j+1
            k = k+1
            yield (*arr,)
        yield (*arr,)
    yield (*arr,)


def quickSort(arr, left, right):
    yield arr
    if left < right and len(arr) > 1:
        pivotindex = int(partition(arr, left, right))
        yield from quickSort(arr, left, pivotindex - 1)
        yield from quickSort(arr, pivotindex + 1, right)
    yield (*arr,)


def partition(arr, left, right):
    i = left
    j = right - 1
    pivot = arr[right]

    while i < j:
        while i < right and arr[i] < pivot:
            i += 1
        while j > left and arr[j] >= pivot:
            j -= 1
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]
    if arr[i] > pivot:
        arr[i], arr[right] = arr[right], arr[i]

    return i


col1 = st.sidebar
col2, col3 = st.columns((2,1))

# ------------------------------- #
# Sidebar + Main panel

col1.header("Calabi Yao ")
option = col1.selectbox("Manifolds", ('Chart', 'Blockchain Explorer', 'Cryptocurrency Top 100', 'Sort Visualizations'))

if option == 'Chart':
    st.header("2Py Lab")
    st.write("Having spent the better part of my experience in Systems Integration and Operations Engineering, I entered a Python Bootcamp in August of 2021 and decided to create this page to display some of the coding skills I've learned in just three months.")
    
    st.write("Here are a few examples of Python programming with a basic Stock Ticker chart lookup tool, a Cryptocurrency Top 100 lookup table by marketcap and Percent Change chart, a Sort Algorythm Visualizer using Matplotlib for data analysis, and a basic (and evolving) Blockchain Block Explorer.")
    sym = col1.text_input("Enter Stock Ticker Symbol:", "ETH-USD", max_chars=None)

    expand = st.expander("About")
    expand.markdown("""
    * **Python Libraries:** streamlit, streamlit.components, pandas, requests, 
    matplotlib, matplotlib.animation, time, random, json, plotly.
    * ** Data sources:** [Nomics.com] (https://nomics.com), [Getblock.io] (https://getblock.io), [Coinbase Pro] (https://pro.cloud.coinbase.com), [Yahoo Finance] (https://yahoo.com/finance).
    * ** APIs:** rpc/application, [Rosetta] (https://www.rosetta-api.org/docs/BlockApi.html) API, [XRP Ledger API] (https://xrpl.org/).
    * ** Layout:** Thanks to [Data Professor] (https://www.youtube.com/channel/UCV8e2g4IWQqK71bbzGDEI4Q0) for 
     streamlit tips and tricks.
    * ** Authored by:** Marx Njoroge, ©2021.
     """)

    cb_api_url = "https://api.pro.coinbase.com"
    bar_size = 3600
    timeend = datetime.now()
    delta = timedelta(hours=1)
    timestart = timeend - (168 * delta)

    timeend = timeend.isoformat()
    timestart = timestart.isoformat()

    params = {
        "start": timestart,
        "end": timeend,
        "granularity": bar_size,
    }
    cb_headers = {"Accept": "application/json"}

    cb_data = rq.get(f"{cb_api_url}/products/{sym}/candles",
                     json=params,
                     headers=cb_headers)
    st.subheader(sym.upper())
    padding = 2

    minselect = col1.select_slider("Time Delta", ["2min", "3min", "5min", "15min", "30min", "60min", "240min"])

    df = pd.DataFrame(cb_data.json(),
                      columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df = df[['date', 'low', 'high', 'open', 'close', 'volume']]
    df.set_index("date", inplace=True)
    df = df.resample(minselect).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "max"
    })
    df.reset_index("date", inplace=True)

    fig = go.Figure(data=[go.Candlestick(x=df["date"],
                                         open=df["open"],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'],
                                         name=sym,
                                         increasing_line_color='magenta',
                                         decreasing_line_color='lightgrey'
                                         )])
    fig.update_xaxes(type='category')
    fig.update_layout(height=600, width=800)

    st.plotly_chart(fig, use_container_width=True)

    st.write("Using Coinbase Pro's API to populate OHLC data, this chart uses Plotly's Graph Objects library to render the interactive candlestick chart from parsed a Pandas DataFrame.")

    st.dataframe(df)
    # st.write(df)

    # tickerData = yf.Ticker(symbol)
    # tickerDf = tickerData.history(period='ytd', interval='1d')
    #
    # st.line_chart(tickerDf.Close)
    # st.image(f"https://finviz.com/chart.ashx?t={symbol}")

if option == 'Cryptocurrency Top 100':

    with st.container():
        st.header(option)
        st.subheader("Python Web Scraping + API + Pandas + Matplotlib")
        st.write("It's fun when a project becomes daily useful. What began as a web scraping exercise evolved into an at-a-glance "
                   "cryptocurrency price change chart.  Web scraping is used to provide the top 100 currency list, which is then fed back to "
                   "the Nomic API to retrieve spot price data and organized into a Pandas DataFrame for the neatly presented tabular data as it "
                   "appears in both the terminal and in the Webified Streamlit App page.  Finally Matplotlib provides a handy way to visualize the "
                   "tabular data in a convenient bar chart that has become a valuable reference for Crypto performance across currencies.")
        st.write("This tool has become a 'go to' screen and plans are to extend this page as the basis for a more expansive dashboard")
    
    # @st.cache
    def load_data():

        nom_url = "https://nomics.com/"
        nom_api_url = "https://api.nomics.com/v1/currencies/ticker"

        request = rq.get(nom_url)

        soup = bs(request.content, 'html.parser')
        nom_data = soup.find('script', id="__NEXT_DATA__", type="application/json")
        # print(nom_data.prettify())

        coins = []

        coin_data = json.loads(nom_data.contents[0])
        listings = coin_data['props']['pageProps']['data']['currenciesTicker']
        for i in listings:
            coins.append(i['id'])

        coin = ','.join(coins)
        
        # st.write("DB username:", st.secrets["db_username"])        
        NOM_API_KEY = st.secrets["NOM_API_KEY"]
        nom_headers = {
            "key": "NOM_API_KEY"
        }
        nom_params = {
            "ids": coin,
            "interval": "1d,30d",
            "convert": "USD"
        }
        url = f"https://api.nomics.com/v1/currencies/ticker?key={NOM_API_KEY}&ids={coin}&interval=1d,30d&convert=USD"
        data = rq.post(url=url).json()

        # print(data[0])

        market_cap = []
        volume = []
        price = []
        price_change_pct_1d = []
        price_change_pct_30d = []
        price_timestamp = []
        name = []
        symbol = []

        for item in data:
            market_cap.append(int(item['market_cap']))
            volume.append(item['1d']['volume'])
            price.append(item['price'])
            price_change_pct_1d.append(float(item['1d']['price_change_pct']))
            price_change_pct_30d.append(float(item['30d']['price_change_pct']))
            price_timestamp.append(item['price_timestamp'])
            name.append(item['name'])
            symbol.append(item['symbol'])

        df = pd.DataFrame(columns=['name', 'symbol', 'price', 'price_change_pct_1d', 'price_timestamp', 'volume', 'marketcap'])
        df['marketcap'] = market_cap
        df['volume'] = volume
        df['price'] = price
        df['price_change_pct_1d'] = price_change_pct_1d
        df['price_change_pct_30d'] = price_change_pct_30d
        df['price_timestamp'] = price_timestamp
        df['name'] = name
        df['symbol'] = symbol

        # df.to_csv("cryptos.csv", index=False)
        print(name)

        return df


    col4, col5 = st.columns((2, 1))
    with col4: st.subheader("CryptoWatch: (%) Price Change")
    period = st.sidebar.selectbox("Time Period", ('1D', '30D'))

    frame = load_data()
    with col4: st.dataframe(frame)

    df_change = pd.concat([frame.symbol, frame.price_change_pct_1d, frame.price_change_pct_30d], axis=1)
    df_change = df_change.set_index('symbol')
    df_change['positive_price_change_pct_1d'] = df_change['price_change_pct_1d'] > 0
    df_change['positive_price_change_pct_30d'] = df_change['price_change_pct_30d'] > 0
    with col4: st.dataframe(df_change)

    if period == '1D':
        with col5: st.subheader("1 Day (%) Price Change")
        df_change = df_change.sort_values(by=['price_change_pct_1d'])
        # with _lock:
        plt.figure(figsize=(6, 20))
        plt.subplots_adjust(top=0.75, bottom=0)
        df_change['price_change_pct_1d'].plot(kind='barh', color=df_change.positive_price_change_pct_1d.map({True: 'purple', False: 'gray'}))
        with col5: st.pyplot(plt)

    elif period == '30D':
        with col5: st.subheader("30 Day (%) Price Change")
        df_change = df_change.sort_values(by=['price_change_pct_30d'])
        plt.figure(figsize=(6, 20))
        plt.subplots_adjust(top=.75, bottom=0)
        df_change['price_change_pct_30d'].plot(kind='barh',
                                              color=df_change.positive_price_change_pct_30d.map({True: 'purple', False: 'gray'}))
        with col5: st.pyplot(plt)

# if option == 'Crypto Chart':
#
#     nom_url = "https://nomics.com/"
#     request = rq.get(nom_url)
#
#     soup = bs(request.content, 'html.parser')
#     nom_data = soup.find('script', id="__NEXT_DATA__", type="application/json")
#     # print(nom_data.prettify())
#
#     coins = {}
#
#     coin_data = json.loads(nom_data.contents[0])
#     listings = coin_data['props']['pageProps']['data']['currenciesTicker']
#
#     for i in listings:
#         coins[i['id']] = i['name']
#
#     # print(coins)
#     coin = col1.text_input("Coin", 'ETH', max_chars=5)
#     print(coin)
#     nom_url = f"https://nomics.com/assets/{coin}-{coins[coin]}/history"
#     data = rq.get(nom_url)
#     print(data)
#
#     name = []
#     symbol = []
#     date = []
#     open = []
#     high = []
#     low = []
#     close = []
#     volume = []
#
#     for item in data:
#
#         name.append(item['name'])
#         symbol.append(item['symbol'])
#         open.append(item['open'])
#         high.append(float(item['high']))
#         low.append(float(item['low']))
#         close.append(item['close'])
#         volume.append(item['1d']['volume'])
#
#     df = pd.DataFrame(columns=['name', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
#     df['name'] = name
#     df['symbol'] = symbol
#     df['open'] = open
#     df['high'] = high
#     df['low'] = low
#     df['close'] = close
#     df['volume'] = volume
#
#     # df.to_csv("cryptos.csv", index=False)
#
#     print(df)


if option == 'Sort Visualizations':
    col2.header(option)
    title = st.sidebar.radio(label="Sort Algorithms", options=["Merge", "Quick", "Bubble"])

    if title == 'Merge':

        col2.subheader(title)

        st.write("This visualization is written in Python using Matplotlib "
                 "to both visualize and animate the Sort Algorithm.  A Streamlit "
                 "component is then used to dynamically convert the Matplotlib animation "
                 "to javascript in order to render it to html.")
        st.write("**Note:** sorting more values takes longer to render.")

        

        n = st.slider(label="Values", min_value=15, max_value=50)
        alg = 2
        cache = n * 10
        title = "Merge Sort"
        array = [i + 1 for i in range(n)]
        random.shuffle(array)
        algo = mergeSort(array)

        # Initialize fig
        plt.rcParams["figure.figsize"] = (7, 4)
        plt.rcParams["font.size"] = 8
        # with _lock:
        fig, ax = plt.subplots()
        ax.set_title(title)

        bar_rec = ax.bar(range(len(array)), array, align='edge')

        ax.set_xlim(0, n)
        ax.set_ylim(0, int(n * 1.06))

        text = ax.text(0.02, 0.95, "0", transform=ax.transAxes)

        epochs = [0]


        def init():
            ax.bar(range(len(array)), array, align='edge')

        # @st.cache
        def update_plot(array, rect, epochs):
            for rect, val in zip(rect, array):
                rect.set_height(val)
                rect.set_color("#cc00cc")
            text.set_text("No. of operations: {}".format(epochs[0]))
            epochs[0] += 1

            return bar_rec,


        anima = anim.FuncAnimation(fig, update_plot, fargs=(bar_rec, epochs), frames=algo, save_count=cache, interval=20,
                                       repeat=False)
        # plt.show()
        # st.pyplot(plt)

        components.html(anima.to_jshtml(), height=1000)

    if title == 'Quick':
        st.subheader(title)

        st.write("This visualization is written in Python using Matplotlib "
                 "to both visualize and animate the Sort Algorithm.  A Streamlit "
                 "component is then used to dynamically convert the Matplotlib animation "
                 "to javascript in order to render it to html.")
        st.write("**Note:** sorting more values takes longer to render.")
     
        n = st.slider(label="Values", min_value=15, max_value=50)
        alg = 3
        cache = 500
        title = "Quick Sort"
        array = [i + 1 for i in range(n)]
        random.shuffle(array)
        algo = quickSort(array, 0, len(array) - 1)

        # Initialize fig
        # with _lock:
        plt.rcParams["figure.figsize"] = (7, 4)
        plt.rcParams["font.size"] = 8
        fig, ax = plt.subplots()
        ax.set_title(title)

        bar_rec = ax.bar(range(len(array)), array, align='edge')

        ax.set_xlim(0, n)
        ax.set_ylim(0, int(n * 1.06))

        text = ax.text(0.02, 0.95, "0", transform=ax.transAxes)

        epochs = [0]


        def init():
            ax.bar(range(len(array)), array, align='edge', color="#0033ff")

        # @st.cache
        def update_plot(array, rect, epochs):
            for rect, val in zip(rect, array):
                rect.set_height(val)
                rect.set_color("#33cccc")
            text.set_text("No. of operations: {}".format(epochs[0]))
            epochs[0] += 1

            return bar_rec,


        anima = anim.FuncAnimation(fig, update_plot, fargs=(bar_rec, epochs), frames=algo, save_count=cache, interval=20,
                                   repeat=False)
        # plt.show()
        # st.pyplot(plt)

        components.html(anima.to_jshtml(), height=1000)

    if title == 'Bubble':
        st.subheader(title)

        st.write("This visualization is written in Python using Matplotlib "
                 "to both visualize and animate the Sort Algorithm.  A Streamlit "
                 "component is then used to dynamically convert the Matplotlib animation "
                 "to javascript in order to render it to html.")
        st.write("** Note:** sorting more values takes longer to render.")
       
        n = st.slider(label="Values", min_value=15, max_value=50)
        alg = 1
        cache = n * (n**1/2)
        title = "Bubble Sort"
        array = [i + 1 for i in range(n)]
        random.shuffle(array)
        algo = bubbleSort(array)

        # Initialize fig
        # with _lock:
        plt.rcParams["figure.figsize"] = (7, 4)
        plt.rcParams["font.size"] = 8

        fig, ax = plt.subplots()
        ax.set_title(title)

        bar_rec = ax.bar(range(len(array)), array, align='edge', color="#cccccc")

        ax.set_xlim(0, n)
        ax.set_ylim(0, int(n * 1.06))

        text = ax.text(0.02, 0.95, "0", transform=ax.transAxes)

        epochs = [0]

        def init():
            ax.bar(range(len(array)), array, align='edge', color="#00cccc")


        # @st.cache
        def update_plot(array, rect, epochs):
            for rect, val in zip(rect, array):
                rect.set_height(val)
                rect.set_color("#cc00cc")
            text.set_text("No. of operations: {}".format(epochs[0]))
            epochs[0] += 1

            return bar_rec,


        anima = anim.FuncAnimation(fig, update_plot, fargs=(bar_rec, epochs), frames=algo, save_count=cache,
                                   interval=20,
                                   repeat=False)
        # plt.show()
        # st.pyplot(plt)

        components.html(anima.to_jshtml(), height=800)

if option == 'Blockchain Explorer':
    col2.header(option)
    title = st.sidebar.radio(label="Latest Blocks from:", options=["Bitcoin: BTC", "Ethereum: ETH", "Binance Smart Chain: BNB", "Cardano: ADA", "Ripple: XRP"])

    if title == "Bitcoin: BTC":

        col2.image("btclogotop.svg")
        col2.subheader("Bitcoin RPC API")
        st.write("""Using Getblock's Blockchain Node Provider as a gateway to various chains presents different 
                   access methods to each chain's network and data.  The data pulled from the Bitcoin network 
                   below uses the json/rpc API generalized method for HTTPD POST style API calls to the chain.""")  
        st.write("""Posted below is the parsed data in a more readable table of 'N' latest blocks, and the raw json for the latest block contents.""")
                     
        st.write("""It should be noted that even given a standardized API call method, the parameters for each network
                    are chain-specific.""")

        btc_status_endpoint = "https://btc.getblock.io/mainnet/"
        headers = {
            "X-API-KEY": st.secrets["GETBLOCK_API_KEY"]
        }


        def get_latest_block():

            btc_status_params = {
                "jsonrpc": "1.0",
                "id": "bitcoin",
                "method": "getblockchaininfo"
            }

            btc_chaindata = rq.post(url=btc_status_endpoint, json=btc_status_params, headers=headers).json()
            blockhash = btc_chaindata['result']['bestblockhash']

            return blockhash


        latest_blockhash = get_latest_block()

        btc_hash = latest_blockhash
        lastn = []

        index = []
        time = []
        blockhash = []
        blocksize = []
        nTx = []

        for i in range(0, 10):
            last_block_params = {
                "jsonrpc": "1.0",
                "id": "bitcoin",
                "method": "getblock",
                "params": [btc_hash]
            }
            new_block = {}
            btc_blockdata = rq.post(url=btc_status_endpoint, json=last_block_params, headers=headers).json()
            new_block['index'] = btc_blockdata['result']['height']
            new_block['timestamp'] = btc_blockdata['result']['time']
            new_block['blockhash'] = btc_blockdata['result']['hash']
            new_block['blocksize'] = btc_blockdata['result']['size']
            new_block['nTx'] = btc_blockdata['result']['nTx']
            index.append(new_block['index'])
            time.append(datetime.fromtimestamp(new_block['timestamp']).strftime('%Y.%m.%d %H:%M:%S'))
            blockhash.append(new_block['blockhash'])
            blocksize.append(new_block['blocksize'])
            nTx.append(new_block['nTx'])

            lastn.append(new_block)
            btc_hash = btc_blockdata['result']['previousblockhash']

        df = pd.DataFrame(columns=['index', 'timestamp (utc)', 'blockhash', 'blocksize', 'no.transactions'])
        df['index'] = index
        df['timestamp (utc)'] = time
        df['blockhash'] = blockhash
        df['blocksize'] = blocksize
        df['no.transactions'] = nTx

        st.dataframe(df)
        st.json(lastn)

    if title == "Ethereum: ETH":

        col2.image("ethereum-logo-landscape-black.png")
        col2.subheader("Ethereum RPC API")
        st.write("""Using Getblock's Blockchain Node Provider as a gateway to various chains presents different access methods to each 
        chain's network and data.  The data pulled from the Bitcoin network below uses the json/rpc API generalized method for rpc/application 
        POST API calls to the chain.""")
        st.write("""Posted below is the parsed data in more readable table of 'N' latest blocks and the raw json for the latest block contents.""")
        st.write("""The Ehereum blockchain uses it's own RPC API call method, the parameters for each network remaining chain-specific.""")

        GETBLOCK_API_URL = "https://eth.getblock.io/mainnet/"
        GETBLOCK_API_KEY = st.secrets["GETBLOCK_API_KEY"]

        eth_headers = {
            "X-API-KEY": GETBLOCK_API_KEY
        }

        blocknumber = []
        timestamp = []
        blockhash = []
        gasLimit = []
        gasUsed = []

        latest_blocks = []
        block_num = "latest"
        eth_params = {
            "id": "blockNumber",
            "jsonrpc": "2.0",
            "method": "eth_getBlockByNumber",
            "params": [block_num, False]
        }
        eth_blockdata = rq.post(url=GETBLOCK_API_URL, json=eth_params, headers=eth_headers).json()
        curr_block_hash = eth_blockdata['result']['hash']
        curr_block_number = eth_blockdata['result']['number']

        for i in range(0, 10):
            hash_eth_params = {
                "id": "blockHash",
                "jsonrpc": "2.0",
                "method": "eth_getBlockByHash",
                "params": [curr_block_hash, False]
            }
            hash_eth_blockdata = rq.post(url=GETBLOCK_API_URL, json=hash_eth_params, headers=eth_headers).json()
            new_block = {'blocknumber': hash_eth_blockdata['result']['number'],
                         'timestamp': hash_eth_blockdata['result']['timestamp'],
                         'blockhash': hash_eth_blockdata['result']['hash'],
                         'gasLimit': hash_eth_blockdata['result']['gasLimit'],
                         'gasUsed': hash_eth_blockdata['result']['gasUsed']}

            latest_blocks.append(new_block)
            curr_block_hash = hash_eth_blockdata['result']['parentHash']

        dec_blocks = []
        for dc in latest_blocks:
            dc['blocknumber'] = int(dc['blocknumber'], 16)
            dc['timestamp'] = int(dc['timestamp'], 16)
            dc['gasLimit'] = int(dc['gasLimit'], 16)
            dc['gasUsed'] = int(dc['gasUsed'], 16)
            blocknumber.append(dc['blocknumber'])
            timestamp.append(datetime.fromtimestamp(dc['timestamp']).strftime('%Y.%m.%d %H:%M:%S'))
            blockhash.append(dc['blockhash'])
            gasLimit.append(dc['gasLimit'])
            gasUsed.append(dc['gasUsed'])

            dec_blocks.append(dc)

        df = pd.DataFrame(columns=['blocknumber', 'timestamp (utc)', 'blockhash', 'gasLimit', 'gasUsed'])
        df['blocknumber'] = blocknumber
        df['timestamp (utc)'] = timestamp
        df['blockhash'] = blockhash
        df['gasLimit'] = gasLimit
        df['gasUsed'] = gasUsed

        st.dataframe(df)
        st.json(dec_blocks)

    if title == "Cardano: ADA":

        col2.image("cardanosizedlogo.svg")
        st.subheader("Cardano [Rosetta] (https://www.rosetta-api.org/docs/BlockApi.html) API.")
        st.write("""Using Getblock's Blockchain Node Provider as a gateway to various chains presents different
                   access methods to each chain's network and data.  The data pulled from the Cardano network
                   below uses the Rosetta API call.""")

        ada_status_endpoint = "https://ada.getblock.io/mainnet/network/status"
        GETBLOCK_API_KEY = st.secrets["GETBLOCK_API_KEY"]
        headers = {
            "X-API-KEY": GETBLOCK_API_KEY,
            "Content-Type": "application/json"
        }

        status_params = {
            "network_identifier": {
                "blockchain": "cardano",
                "network": "mainnet"},
            "metadata": {}
        }

        ada_status = rq.post(url=ada_status_endpoint, json=status_params, headers=headers).json()
        # pprint(ada_status)
        curr_block_idx = ada_status['current_block_identifier']['index']
        curr_block_hash = ada_status['current_block_identifier']['hash']
        latest_blocks = []

        epoch = []
        index = []
        timestamp = []
        blockhash = []
        blocksize = []

        for i in range(0, 10):
            ada_block_endpoint = "https://ada.getblock.io/mainnet/block"
            block_params = {
                "network_identifier": {
                    "blockchain": "cardano",
                    "network": "mainnet"},
                "metadata": {},
                "block_identifier": {
                    "index": curr_block_idx,
                    "hash": curr_block_hash}
            }

            block_data = rq.post(url=ada_block_endpoint, json=block_params, headers=headers).json()
            new_block = {'epoch': block_data['block']['metadata']['epochNo'],
                         'index': block_data['block']['block_identifier']['index'],
                         'timestamp': block_data['block']['timestamp'],
                         'blockhash': block_data['block']['block_identifier']['hash'],
                         'blocksize': block_data['block']['metadata']['size']}
            epoch.append(new_block['epoch'])
            index.append(new_block['index'])
            timestamp.append(datetime.fromtimestamp((new_block['timestamp'])/1000).strftime('%Y.%m.%d %H:%M:%S'))
            blockhash.append(new_block['blockhash'])
            blocksize.append(new_block['blocksize'])

            latest_blocks.append(new_block)

            curr_block_idx = block_data['block']['parent_block_identifier']['index']
            curr_block_hash = block_data['block']['parent_block_identifier']['hash']

        df = pd.DataFrame(columns=['epoch', 'index', 'timestamp (utc)', 'blockhash', 'blocksize'])
        df['epoch'] = epoch
        df['index'] = index
        df['timestamp (utc)'] = timestamp
        df['blockhash'] = blockhash
        df['blocksize'] = blocksize

        st.dataframe(df)
        st.json(latest_blocks)

    if title == "Binance Smart Chain: BNB":

        col2.image("330px-Binance_logo.svg.png")
        col2.subheader(f"{title}: Ethereum RPC API")
        st.write(f"Using Getblock's Blockchain Node Provider as a gateway to various chains presents different access methods to each \
        chain's network and data.  The data pulled from the {title} network below uses the json/rpc API generalized method for rpc/application \
        POST API calls to the chain.")
        st.write(
            """Posted below is the parsed data in more readable table of 'N' latest blocks and the raw json for the latest block contents.""")
        st.write(
            f"The {title} blockchain uses Ethereum RPC API call methods, the parameters for each network remaining chain-specific.")

        GETBLOCK_API_URL = "https://bsc.getblock.io/mainnet/"
        GETBLOCK_API_KEY = st.secrets["GETBLOCK_API_KEY"]

        bsc_headers = {
            "X-API-KEY": GETBLOCK_API_KEY,
            "Content-Type": "application/json"
        }

        blocknumber = []
        timestamp = []
        size = []
        blockhash = []
        gasLimit = []
        gasUsed = []
        miner = []

        latest_blocks = []
        curr_block_num = "latest"

        num_bsc_params = {
            "id": "binance",
            "jsonrpc": "2.0",
            "method": "eth_getBlockByNumber",
            "params": [curr_block_num, True]
        }
        num_bsc_blockdata = rq.post(url=GETBLOCK_API_URL, json=num_bsc_params, headers=bsc_headers).json()

        curr_block_hash = num_bsc_blockdata['result']['hash']
        # print(curr_block_hash)

        for i in range(0, 10):
            hash_bsc_params = {
                "id": "binance",
                "jsonrpc": "2.0",
                "method": "eth_getBlockByHash",
                "params": [curr_block_hash, True]
            }

            hash_bsc_blockdata = rq.post(url=GETBLOCK_API_URL, json=hash_bsc_params, headers=bsc_headers).json()
            # print(hash_bsc_blockdata)
            new_block = {'blocknumber': hash_bsc_blockdata['result']['number'],
                         'timestamp': hash_bsc_blockdata['result']['timestamp'],
                         'size': hash_bsc_blockdata['result']['size'],
                         'blockhash': hash_bsc_blockdata['result']['hash'],
                         'gasLimit': hash_bsc_blockdata['result']['gasLimit'],
                         'gasUsed': hash_bsc_blockdata['result']['gasUsed'],
                         'miner': hash_bsc_blockdata['result']['miner']}

            latest_blocks.append(new_block)
            curr_block_hash = hash_bsc_blockdata['result']['parentHash']

        dec_blocks = []
        for dc in latest_blocks:
            dc['blocknumber'] = int(dc['blocknumber'], 16)
            dc['timestamp'] = int(dc['timestamp'], 16)
            dc['size'] = int(dc['size'], 16)
            dc['gasLimit'] = int(dc['gasLimit'], 16)
            dc['gasUsed'] = int(dc['gasUsed'], 16)
            blocknumber.append(dc['blocknumber'])
            timestamp.append(datetime.fromtimestamp(dc['timestamp']).strftime('%Y.%m.%d %H:%M:%S'))
            size.append(dc['size'])
            blockhash.append(dc['blockhash'])
            gasLimit.append(dc['gasLimit'])
            gasUsed.append(dc['gasUsed'])
            miner.append(dc['miner'])

            dec_blocks.append(dc)

        df = pd.DataFrame(columns=['blocknumber', 'timestamp', 'size', 'blockhash', 'gasLimit', 'gasUsed', 'miner'])
        df['blocknumber'] = blocknumber
        df['timestamp'] = timestamp
        df['size'] = size
        df['blockhash'] = blockhash
        df['gasLimit'] = gasLimit
        df['gasUsed'] = gasUsed
        df['miner'] = miner

        st.dataframe(df)
        st.json(dec_blocks)

    if title == "Ripple: XRP":

        col2.subheader("XRP Ledger API")
        st.image("ripplesmallimage.png")
        st.write("""Ripple provides it's own gateways to the XRP chains, which present APIs to their unique syntax on their own
        proprietary network and protocolsto provide access to each chain's network and data.  The data pulled from the Ripple network 
        below uses the json/rpc API generalized method for HTTPD POST style API calls to the chain.""")
        st.write(
            """Posted below is the parsed data in a more readable table of 'N' latest blocks, and the raw json for the latest block contents.""")

        st.write("""It should be noted that even given a standardized API call method, the parameters for each network
                            are chain-specific.""")

        XRP_API_URL = "https://s1.ripple.com:51234/"
        xrp_headers = {
            "content-type": "application/json"
        }

        blocknumber = []
        timestamp = []
        block_hash = []
        total_coins = []
        transaction_hash = []

        latest_blocks = []
        latest_tx = []

        init_ledger_hash = "validated"
        METHOD = "ledger_closed"

        for i in range(0, 1):
            xrp_params = {
                "method": METHOD,
                "params": [
                    {
                        "ledger_index": init_ledger_hash
                    }
                ]
            }
            xrp_data = rq.post(url=XRP_API_URL, json=xrp_params, headers=xrp_headers).json()
            curr_blockhash = xrp_data['result']['ledger_hash']
            curr_blockindex = xrp_data['result']['ledger_index']
            # pprint(xrp_data)

        METHOD = "ledger_data"

        for i in range(0, 10):
            xrp_params = {
                "method": METHOD,
                "params": [
                    {
                        "binary": False,
                        "ledger_hash": curr_blockhash,
                        "limit": 10
                    }
                ]
            }
            xrp_data = rq.post(url=XRP_API_URL, json=xrp_params, headers=xrp_headers).json()

            blocknumber.append(xrp_data['result']['ledger']['ledger_index'])
            timestamp.append(xrp_data['result']['ledger']['close_time_human'])
            block_hash.append(xrp_data['result']['ledger']['hash'])
            total_coins.append(xrp_data['result']['ledger']['total_coins'])
            transaction_hash.append(xrp_data['result']['ledger']['transaction_hash'])

            curr_blockhash = xrp_data['result']['ledger']['parent_hash']

        # pprint(xrp_data)

        st.write("Latest Ledgers (Blocks)")

        df = pd.DataFrame(columns=['blocknumber', 'timestamp', 'block_hash', 'total_coins', 'transaction_hash'])
        df['blocknumber'] = blocknumber
        df['timestamp'] = timestamp
        df['block_hash'] = block_hash
        df['total_coins'] = total_coins
        df['transaction_hash'] = transaction_hash

        st.dataframe(df)

        st.write("Latest Transactions")

        txledger_index = []
        trans_type = []
        account = []
        fee = []
        expiration = []
        tx_hash = []

        METHOD = "tx_history"

        xrp_tx_params = {
            "method": METHOD,
            "params": [
                {
                    "start": 0
                }
            ]
        }
        xrp_tx_data = rq.post(url=XRP_API_URL, json=xrp_tx_params, headers=xrp_headers).json()
        latest_tx.append(xrp_tx_data['result']['txs'])

        for i in range(0, len(latest_tx[0])):
            txledger_index.append(latest_tx[0][i]['ledger_index'])
            trans_type.append(latest_tx[0][i]['TransactionType'])
            account.append(latest_tx[0][i]['Account'])
            fee.append(latest_tx[0][i]['Fee'])
            #     expiration.append(latest_tx[0][tx]['Expiration'])
            tx_hash.append(latest_tx[0][i]['hash'])

        txdf = pd.DataFrame(columns=['txledger_index', 'trans_type', 'account', 'fee', 'tx_hash'])
        txdf['txledger_index'] = txledger_index
        txdf['trans_type'] = trans_type
        txdf['account'] = account
        txdf['fee'] = fee
        # txdf['expiration'] = expiration
        txdf['tx_hash'] = tx_hash

        st.dataframe(txdf)
