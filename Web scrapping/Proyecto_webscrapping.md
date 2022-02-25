# Webscrapping

To extract data from websites,  we can use the **read_html** function to directly get DataFrames from a **url**. 


```python
import pandas as pd
url = "https://en.wikipedia.org/wiki/World_population"
dataframe_list = pd.read_html(url, flavor='bs4')
display(dataframe_list[5])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rank</th>
      <th>Country</th>
      <th>Population</th>
      <th>Area(km2)</th>
      <th>Density(pop/km2)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Singapore</td>
      <td>5704000</td>
      <td>710</td>
      <td>8033</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Bangladesh</td>
      <td>172240000</td>
      <td>143998</td>
      <td>1196</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Palestine</td>
      <td>5266785</td>
      <td>6020</td>
      <td>847</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Lebanon</td>
      <td>6856000</td>
      <td>10452</td>
      <td>656</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Taiwan</td>
      <td>23604000</td>
      <td>36193</td>
      <td>652</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>South Korea</td>
      <td>51781000</td>
      <td>99538</td>
      <td>520</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Rwanda</td>
      <td>12374000</td>
      <td>26338</td>
      <td>470</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Haiti</td>
      <td>11578000</td>
      <td>27065</td>
      <td>428</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Netherlands</td>
      <td>17690000</td>
      <td>41526</td>
      <td>426</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Israel</td>
      <td>9480000</td>
      <td>22072</td>
      <td>429</td>
    </tr>
  </tbody>
</table>
</div>


However, sometimes the server refuses to authorize the request **(HTTP Error 403: Forbidden )**.


```python
url= 'https://www.worldometers.info/world-population'
dataframe_list = pd.read_html(url, flavor='bs4')
```


    ---------------------------------------------------------------------------

    HTTPError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_6888/80773953.py in <module>
          1 url= 'https://www.worldometers.info/world-population'
    ----> 2 dataframe_list = pd.read_html(url, flavor='bs4')
    

    ~\miniconda3\lib\site-packages\pandas\util\_decorators.py in wrapper(*args, **kwargs)
        309                     stacklevel=stacklevel,
        310                 )
    --> 311             return func(*args, **kwargs)
        312 
        313         return wrapper
    

    ~\miniconda3\lib\site-packages\pandas\io\html.py in read_html(io, match, flavor, header, index_col, skiprows, attrs, parse_dates, thousands, encoding, decimal, converters, na_values, keep_default_na, displayed_only)
       1096     io = stringify_path(io)
       1097 
    -> 1098     return _parse(
       1099         flavor=flavor,
       1100         io=io,
    

    ~\miniconda3\lib\site-packages\pandas\io\html.py in _parse(flavor, io, match, attrs, encoding, displayed_only, **kwargs)
        904 
        905         try:
    --> 906             tables = p.parse_tables()
        907         except ValueError as caught:
        908             # if `io` is an io-like object, check if it's seekable
    

    ~\miniconda3\lib\site-packages\pandas\io\html.py in parse_tables(self)
        220         list of parsed (header, body, footer) tuples from tables.
        221         """
    --> 222         tables = self._parse_tables(self._build_doc(), self.match, self.attrs)
        223         return (self._parse_thead_tbody_tfoot(table) for table in tables)
        224 
    

    ~\miniconda3\lib\site-packages\pandas\io\html.py in _build_doc(self)
        599         from bs4 import BeautifulSoup
        600 
    --> 601         bdoc = self._setup_build_doc()
        602         if isinstance(bdoc, bytes) and self.encoding is not None:
        603             udoc = bdoc.decode(self.encoding)
    

    ~\miniconda3\lib\site-packages\pandas\io\html.py in _setup_build_doc(self)
        591 
        592     def _setup_build_doc(self):
    --> 593         raw_text = _read(self.io)
        594         if not raw_text:
        595             raise ValueError(f"No text parsed from document: {self.io}")
    

    ~\miniconda3\lib\site-packages\pandas\io\html.py in _read(obj)
        132     """
        133     if is_url(obj):
    --> 134         with urlopen(obj) as url:
        135             text = url.read()
        136     elif hasattr(obj, "read"):
    

    ~\miniconda3\lib\site-packages\pandas\io\common.py in urlopen(*args, **kwargs)
        210     import urllib.request
        211 
    --> 212     return urllib.request.urlopen(*args, **kwargs)
        213 
        214 
    

    ~\miniconda3\lib\urllib\request.py in urlopen(url, data, timeout, cafile, capath, cadefault, context)
        212     else:
        213         opener = _opener
    --> 214     return opener.open(url, data, timeout)
        215 
        216 def install_opener(opener):
    

    ~\miniconda3\lib\urllib\request.py in open(self, fullurl, data, timeout)
        521         for processor in self.process_response.get(protocol, []):
        522             meth = getattr(processor, meth_name)
    --> 523             response = meth(req, response)
        524 
        525         return response
    

    ~\miniconda3\lib\urllib\request.py in http_response(self, request, response)
        630         # request was successfully received, understood, and accepted.
        631         if not (200 <= code < 300):
    --> 632             response = self.parent.error(
        633                 'http', request, response, code, msg, hdrs)
        634 
    

    ~\miniconda3\lib\urllib\request.py in error(self, proto, *args)
        559         if http_err:
        560             args = (dict, 'default', 'http_error_default') + orig_args
    --> 561             return self._call_chain(*args)
        562 
        563 # XXX probably also want an abstract factory that knows when it makes
    

    ~\miniconda3\lib\urllib\request.py in _call_chain(self, chain, kind, meth_name, *args)
        492         for handler in handlers:
        493             func = getattr(handler, meth_name)
    --> 494             result = func(*args)
        495             if result is not None:
        496                 return result
    

    ~\miniconda3\lib\urllib\request.py in http_error_default(self, req, fp, code, msg, hdrs)
        639 class HTTPDefaultErrorHandler(BaseHandler):
        640     def http_error_default(self, req, fp, code, msg, hdrs):
    --> 641         raise HTTPError(req.full_url, code, msg, hdrs, fp)
        642 
        643 class HTTPRedirectHandler(BaseHandler):
    

    HTTPError: HTTP Error 403: Forbidden


To overcome this issue, we can scrape data from HTML tables into a Dataframe using BeautifulSoup


```python
from bs4 import BeautifulSoup # module for web scrapping.
import requests  #  module to download a web page
```


```python
url= 'https://www.worldometers.info/world-population'
data  = requests.get(url).text
soup = BeautifulSoup(data,"html.parser")
```


```python
#find all html tables in the web page
tables = soup.find_all('table')
```


```python
# we can see how many tables were found by checking the length of the tables list
number_of_tables= len(tables)
number_of_tables
```




    5



**Function to get the columns names**


```python
def get_column_names(my_table):
    columns= my_table.findAll('th') 
    columns_name = [x.text.strip() for x in columns] 
    return columns_name
    
```

**Function to create the dataframe corresponding to a table**


```python
def create_dataframe(table):
    columns_name=get_column_names(table)
    df = pd.DataFrame(columns=columns_name)
    for row in table.tbody.find_all("tr"):
        col = row.find_all("td")
        if (col != []):
            my_data = [x.text.strip() for x in col] # List comprehension
            my_data_columns=zip(columns_name, my_data)
            df = df.append(dict(my_data_columns), ignore_index=True)         
    return df
```

**Loop over all tables in the websitet**: create a dataframe for each table and display it. Adittionally, we can save the table in a csv format. If the table has a caption, the csv file will be save with the caption name. Otherwise it will be saved with the name 'tableX'


```python
my_list=[]
for index,table in enumerate(tables):
        table_index = index
        title=table.find('caption')
        if title==None:
            save_name='table_'+ str(table_index)
        else:
            save_name= title.text.strip()
        print(save_name)
        my_df=create_dataframe(table)
        my_list.extend([my_df])
        display(my_df)
        # Save df to csv
        my_df.to_csv(save_name+'.csv')  


```

    table_0
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year (July 1)</th>
      <th>Population</th>
      <th>Yearly %  Change</th>
      <th>Yearly Change</th>
      <th>Median Age</th>
      <th>Fertility Rate</th>
      <th>Density (P/Km²)</th>
      <th>Urban Pop %</th>
      <th>Urban Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018</td>
      <td>7,631,091,040</td>
      <td>1.10 %</td>
      <td>83,232,115</td>
      <td>29.8</td>
      <td>2.51</td>
      <td>51</td>
      <td>55.3 %</td>
      <td>4,219,817,318</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>7,547,858,925</td>
      <td>1.12 %</td>
      <td>83,836,876</td>
      <td>29.8</td>
      <td>2.51</td>
      <td>51</td>
      <td>54.9 %</td>
      <td>4,140,188,594</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>7,464,022,049</td>
      <td>1.14 %</td>
      <td>84,224,910</td>
      <td>29.8</td>
      <td>2.51</td>
      <td>50</td>
      <td>54.4 %</td>
      <td>4,060,652,683</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>7,379,797,139</td>
      <td>1.19 %</td>
      <td>84,594,707</td>
      <td>30</td>
      <td>2.52</td>
      <td>50</td>
      <td>54.0 %</td>
      <td>3,981,497,663</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010</td>
      <td>6,956,823,603</td>
      <td>1.24 %</td>
      <td>82,983,315</td>
      <td>28</td>
      <td>2.58</td>
      <td>47</td>
      <td>51.7 %</td>
      <td>3,594,868,146</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2005</td>
      <td>6,541,907,027</td>
      <td>1.26 %</td>
      <td>79,682,641</td>
      <td>27</td>
      <td>2.65</td>
      <td>44</td>
      <td>49.2 %</td>
      <td>3,215,905,863</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2000</td>
      <td>6,143,493,823</td>
      <td>1.35 %</td>
      <td>79,856,169</td>
      <td>26</td>
      <td>2.78</td>
      <td>41</td>
      <td>46.7 %</td>
      <td>2,868,307,513</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1995</td>
      <td>5,744,212,979</td>
      <td>1.52 %</td>
      <td>83,396,384</td>
      <td>25</td>
      <td>3.01</td>
      <td>39</td>
      <td>44.8 %</td>
      <td>2,575,505,235</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1990</td>
      <td>5,327,231,061</td>
      <td>1.81 %</td>
      <td>91,261,864</td>
      <td>24</td>
      <td>3.44</td>
      <td>36</td>
      <td>43.0 %</td>
      <td>2,290,228,096</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1985</td>
      <td>4,870,921,740</td>
      <td>1.79 %</td>
      <td>82,583,645</td>
      <td>23</td>
      <td>3.59</td>
      <td>33</td>
      <td>41.2 %</td>
      <td>2,007,939,063</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1980</td>
      <td>4,458,003,514</td>
      <td>1.79 %</td>
      <td>75,704,582</td>
      <td>23</td>
      <td>3.86</td>
      <td>30</td>
      <td>39.3 %</td>
      <td>1,754,201,029</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1975</td>
      <td>4,079,480,606</td>
      <td>1.97 %</td>
      <td>75,808,712</td>
      <td>22</td>
      <td>4.47</td>
      <td>27</td>
      <td>37.7 %</td>
      <td>1,538,624,994</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1970</td>
      <td>3,700,437,046</td>
      <td>2.07 %</td>
      <td>72,170,690</td>
      <td>22</td>
      <td>4.93</td>
      <td>25</td>
      <td>36.6 %</td>
      <td>1,354,215,496</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1965</td>
      <td>3,339,583,597</td>
      <td>1.93 %</td>
      <td>60,926,770</td>
      <td>22</td>
      <td>5.02</td>
      <td>22</td>
      <td>N.A.</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1960</td>
      <td>3,034,949,748</td>
      <td>1.82 %</td>
      <td>52,385,962</td>
      <td>23</td>
      <td>4.90</td>
      <td>20</td>
      <td>33.7 %</td>
      <td>1,023,845,517</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1955</td>
      <td>2,773,019,936</td>
      <td>1.80 %</td>
      <td>47,317,757</td>
      <td>23</td>
      <td>4.97</td>
      <td>19</td>
      <td>N.A.</td>
      <td>N.A.</td>
    </tr>
  </tbody>
</table>
</div>


    table_1
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year (July 1)</th>
      <th>Population</th>
      <th>Yearly %  Change</th>
      <th>Yearly Change</th>
      <th>Median  Age</th>
      <th>Fertility  Rate</th>
      <th>Density  (P/Km²)</th>
      <th>Urban Pop %</th>
      <th>Urban Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020</td>
      <td>7,794,798,739</td>
      <td>1.10 %</td>
      <td>83,000,320</td>
      <td>31</td>
      <td>2.47</td>
      <td>52</td>
      <td>56.2 %</td>
      <td>4,378,993,944</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2025</td>
      <td>8,184,437,460</td>
      <td>0.98 %</td>
      <td>77,927,744</td>
      <td>32</td>
      <td>2.54</td>
      <td>55</td>
      <td>58.3 %</td>
      <td>4,774,646,303</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2030</td>
      <td>8,548,487,400</td>
      <td>0.87 %</td>
      <td>72,809,988</td>
      <td>33</td>
      <td>2.62</td>
      <td>57</td>
      <td>60.4 %</td>
      <td>5,167,257,546</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2035</td>
      <td>8,887,524,213</td>
      <td>0.78 %</td>
      <td>67,807,363</td>
      <td>34</td>
      <td>2.70</td>
      <td>60</td>
      <td>62.5 %</td>
      <td>5,555,833,477</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2040</td>
      <td>9,198,847,240</td>
      <td>0.69 %</td>
      <td>62,264,605</td>
      <td>35</td>
      <td>2.77</td>
      <td>62</td>
      <td>64.6 %</td>
      <td>5,938,249,026</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2045</td>
      <td>9,481,803,274</td>
      <td>0.61 %</td>
      <td>56,591,207</td>
      <td>35</td>
      <td>2.85</td>
      <td>64</td>
      <td>66.6 %</td>
      <td>6,312,544,819</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2050</td>
      <td>9,735,033,990</td>
      <td>0.53 %</td>
      <td>50,646,143</td>
      <td>36</td>
      <td>2.95</td>
      <td>65</td>
      <td>68.6 %</td>
      <td>6,679,756,162</td>
    </tr>
  </tbody>
</table>
</div>


    table_2
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
    </tr>
  </tbody>
</table>
</div>


    table_3
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>Region</th>
      <th>Population(2020)</th>
      <th>YearlyChange</th>
      <th>NetChange</th>
      <th>Density(P/Km²)</th>
      <th>Land Area(Km²)</th>
      <th>Migrants(net)</th>
      <th>Fert.Rate</th>
      <th>Med.Age</th>
      <th>UrbanPop %</th>
      <th>WorldShare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Asia</td>
      <td>4,641,054,775</td>
      <td>0.86 %</td>
      <td>39,683,577</td>
      <td>150</td>
      <td>31,033,131</td>
      <td>-1,729,112</td>
      <td>2.2</td>
      <td>32</td>
      <td>0 %</td>
      <td>59.5 %</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Africa</td>
      <td>1,340,598,147</td>
      <td>2.49 %</td>
      <td>32,533,952</td>
      <td>45</td>
      <td>29,648,481</td>
      <td>-463,024</td>
      <td>4.4</td>
      <td>20</td>
      <td>0 %</td>
      <td>17.2 %</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Europe</td>
      <td>747,636,026</td>
      <td>0.06 %</td>
      <td>453,275</td>
      <td>34</td>
      <td>22,134,900</td>
      <td>1,361,011</td>
      <td>1.6</td>
      <td>43</td>
      <td>0 %</td>
      <td>9.6 %</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Latin America and the Caribbean</td>
      <td>653,962,331</td>
      <td>0.9 %</td>
      <td>5,841,374</td>
      <td>32</td>
      <td>20,139,378</td>
      <td>-521,499</td>
      <td>2</td>
      <td>31</td>
      <td>0 %</td>
      <td>8.4 %</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Northern America</td>
      <td>368,869,647</td>
      <td>0.62 %</td>
      <td>2,268,683</td>
      <td>20</td>
      <td>18,651,660</td>
      <td>1,196,400</td>
      <td>1.8</td>
      <td>39</td>
      <td>0 %</td>
      <td>4.7 %</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Oceania</td>
      <td>42,677,813</td>
      <td>1.31 %</td>
      <td>549,778</td>
      <td>5</td>
      <td>8,486,460</td>
      <td>156,226</td>
      <td>2.4</td>
      <td>33</td>
      <td>0 %</td>
      <td>0.5 %</td>
    </tr>
  </tbody>
</table>
</div>


    table_4
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>Country (or dependency)</th>
      <th>Population(2020)</th>
      <th>YearlyChange</th>
      <th>NetChange</th>
      <th>Density (P/Km²)</th>
      <th>Land Area (Km²)</th>
      <th>Migrants(net)</th>
      <th>Fert.Rate</th>
      <th>Med.Age</th>
      <th>UrbanPop %</th>
      <th>WorldShare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>China</td>
      <td>1,439,323,776</td>
      <td>0.39 %</td>
      <td>5,540,090</td>
      <td>153</td>
      <td>9,388,211</td>
      <td>-348,399</td>
      <td>1.69</td>
      <td>38</td>
      <td>60.8 %</td>
      <td>18.5 %</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>India</td>
      <td>1,380,004,385</td>
      <td>0.99 %</td>
      <td>13,586,631</td>
      <td>464</td>
      <td>2,973,190</td>
      <td>-532,687</td>
      <td>2.2402</td>
      <td>28</td>
      <td>35 %</td>
      <td>17.7 %</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>United States</td>
      <td>331,002,651</td>
      <td>0.59 %</td>
      <td>1,937,734</td>
      <td>36</td>
      <td>9,147,420</td>
      <td>954,806</td>
      <td>1.7764</td>
      <td>38</td>
      <td>82.8 %</td>
      <td>4.2 %</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Indonesia</td>
      <td>273,523,615</td>
      <td>1.07 %</td>
      <td>2,898,047</td>
      <td>151</td>
      <td>1,811,570</td>
      <td>-98,955</td>
      <td>2.3195</td>
      <td>30</td>
      <td>56.4 %</td>
      <td>3.5 %</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Pakistan</td>
      <td>220,892,340</td>
      <td>2 %</td>
      <td>4,327,022</td>
      <td>287</td>
      <td>770,880</td>
      <td>-233,379</td>
      <td>3.55</td>
      <td>23</td>
      <td>35.1 %</td>
      <td>2.8 %</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>230</th>
      <td>231</td>
      <td>Montserrat</td>
      <td>4,992</td>
      <td>0.06 %</td>
      <td>3</td>
      <td>50</td>
      <td>100</td>
      <td></td>
      <td>N.A.</td>
      <td>N.A.</td>
      <td>9.6 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>231</th>
      <td>232</td>
      <td>Falkland Islands</td>
      <td>3,480</td>
      <td>3.05 %</td>
      <td>103</td>
      <td>0</td>
      <td>12,170</td>
      <td></td>
      <td>N.A.</td>
      <td>N.A.</td>
      <td>66 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>232</th>
      <td>233</td>
      <td>Niue</td>
      <td>1,626</td>
      <td>0.68 %</td>
      <td>11</td>
      <td>6</td>
      <td>260</td>
      <td></td>
      <td>N.A.</td>
      <td>N.A.</td>
      <td>46.4 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>233</th>
      <td>234</td>
      <td>Tokelau</td>
      <td>1,357</td>
      <td>1.27 %</td>
      <td>17</td>
      <td>136</td>
      <td>10</td>
      <td></td>
      <td>N.A.</td>
      <td>N.A.</td>
      <td>0 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>234</th>
      <td>235</td>
      <td>Holy See</td>
      <td>801</td>
      <td>0.25 %</td>
      <td>2</td>
      <td>2,003</td>
      <td>0</td>
      <td></td>
      <td>N.A.</td>
      <td>N.A.</td>
      <td>N.A.</td>
      <td>0 %</td>
    </tr>
  </tbody>
</table>
<p>235 rows × 12 columns</p>
</div>



```python

```
