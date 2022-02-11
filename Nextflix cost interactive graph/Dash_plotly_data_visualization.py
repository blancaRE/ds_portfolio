import pandas as pd
import plotly.express as px  
import plotly.graph_objects as go 
from dash import Dash, dcc, html, Input, Output  
# Import required libraries
import pandas as pd
import dash
#import dash_html_components as html
from dash import html
#import dash_core_components as dcc
from dash import dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from dash import no_update
import seaborn as sns

import os

app = dash.Dash(__name__)

# Clear the layout and do not display exception till callback gets executed
app.config.suppress_callback_exceptions = True


netflix_data =  pd.read_csv('Netflixsubscription.csv', encoding = "ISO-8859-1") # encoding = "ISO-8859-1"


# List of number of countries  
list_countries = [i for i in range(1, netflix_data.shape[0]+1, 1)]

average = netflix_data[['Cost Per Month - Standard ($)','Cost Per Month - Basic ($)', 'Cost Per Month - Premium ($)']].mean(axis=1)
netflix_data.insert(netflix_data.shape[1], 'Average Cost', average)

# Application layout
app.layout = html.Div(children=[ 
                                #  dashboard title
                                html.H1('Netflix Cost per Country', style={'font-family': 'Arial' ,'textAlign':'center', 'color':'Blue', 'font-size':24}),##503D36 

                                html.Div([
                                    html.Div([
                                        html.Div(
                                            [
                                            html.H2('Plan:', style={'font-family': 'Arial','margin-right': '2em'}),
                                            ]
                                        ),
                                        dcc.Dropdown(id='fare-type', 
                                           options=[
                                          {"label": "Basic", "value": 'Cost Per Month - Basic ($)'},  # ve el usuario
                                          {"label": "Standard", "value": 'Cost Per Month - Standard ($)'},  
                                          {"label": "Premium", "value": 'Cost Per Month - Premium ($)'},
                                          {"label": "Average", "value": 'Average Cost'}],
                                          placeholder='Select plan',
                                          style={'width':'80%', 'padding':'3px', 'font-size': '20px', 'text-align-last' : 'center'}),
                                    ], style={'display':'marginBottom'}),

                                   html.Div([
                                       # Create an division for adding dropdown helper text for choosing year
                                        html.Div(
                                            [
                                            html.H2('Choose number of countries in the ranking:', style={'font-family': 'Arial','margin-right': '2em'})
                                            ]
                                        ),                                     
                                       dcc.Slider(
                                            1,
                                            netflix_data.shape[0]+1,
                                            step=1,
                                            id='input-ncountries',
                                            value= netflix_data.shape[0]+1, 
                                            marks={str(year): str(year) for year in list_countries},
                                             ) 
                                           ], style={'display': 'marginBottom'}),   #flex
                                        ]),
                                
                                html.Div([ ], id='plot1'),
    
                                html.Div([
                                        html.Div([ ], id='plot2'),
                                        html.Div([ ], id='plot3')
                                ], style={'display': 'flex'}),
                                    
                                ])

# Callback function definition
@app.callback( [Output(component_id='plot1', component_property='children'),
                Output(component_id='plot2', component_property='children'),
                Output(component_id='plot3', component_property='children')],
               [Input(component_id='fare-type', component_property='value'),
                Input(component_id='input-ncountries', component_property='value')],
               #Holding output state till user enters all the form information. In this case, it will be chart type and year
               [State("plot1", 'children'), State("plot2", "children"),
                State("plot3", "children")])
# Add computation to callback function and return graph
def get_graph(chart, n_countries, c1, c2, c3,):
            ###########Choropleth  MAP

        map_fig = go.Figure(data=go.Choropleth(
            locations = netflix_data['Country'],
            locationmode = 'country names',
            z= netflix_data[chart],   #  z = average, netflix_data['Cost Per Month - Basic ($)']
            colorscale = 'Blues', 
            autocolorscale=False,
            reversescale=True,
            marker_line_color='darkgray',
            marker_line_width=0.5,
            colorbar_tickprefix = '$',
            colorbar_title = 'Cost per month$',
            ))

        map_fig.update_layout(
            title_text='Netflix price',
            geo=dict(
                showframe=False,
                showcoastlines=False,
                projection_type='equirectangular'
            ),
            )
        cost_basic_sort = netflix_data.sort_values(chart, ascending=False).head(n_countries)
        bar_fig = px.bar(cost_basic_sort,  x='Country', y=chart ,title='Top countries')   
        pie_fig = px.pie(cost_basic_sort, values='Total Library Size', names='Country', title= 'Total Library Size')   
        if chart == 'average':

            return [dcc.Graph(figure=map_fig), 
                    dcc.Graph(figure=bar_fig),
                    dcc.Graph(figure=pie_fig)
                   ]
        else:

            
            return[dcc.Graph(figure=map_fig), 
                   dcc.Graph(figure=bar_fig), 
                   dcc.Graph(figure=pie_fig)
#                    dcc.Graph(figure=fig), 
#                    dcc.Graph(figure=fig)
                  ]

if __name__ == '__main__':

    app.run_server()