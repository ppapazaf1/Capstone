from StockPredictor import app
import json, plotly
#from flask import render_template
from flask import Flask, request, url_for, redirect, render_template, Markup
from data_scripts.load_data import return_figures
from data_scripts.load_data import get_stocks

@app.route('/')
def form():
    '''
    Initiate and load home page

    '''

    symbols = get_stocks()

    str1 = '<form action=\"index\">' + \
    '<label for=\"symbols\">Choose a stock symbol:</label>' + \
    '<select name="symbols\" id="symbols">'

    str2 = ""
    # '<option value={{symbols[0]}}>xxx</option>'
    for st in symbols:
        str2 = str2 + '<option value='+ st +'>'+ st +'</option>'
    


    str3 = '</select>' + \
    '<br><br>'

    #+ \
    #'<input type="submit" value="Submit">' + \
    #'</form>'

    stri = Markup(str1 + str2 + str3)
    
    return render_template('home.html', symbols= symbols, stri= stri)
	
	
@app.route('/index',  methods=['GET', 'POST'])
def index():
    '''
    Initiate and load index page that contains graphs and information

    '''
   
    if request.method == 'GET':

        symbol = request.args.get('symbols')
        pred_days = request.args.get('pred_days')

  
        figures, sharp_ratio = return_figures(symbol, pred_days)

        # plot ids for the html id tag
        ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]
	
	    #print(ids)

        # Convert the plotly figures to JSON for javascript in html template
        figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('index.html',
                           ids=ids,
                           figuresJSON=figuresJSON,
						   symbol={symbol}, pred_days={pred_days}, sharpe_ratio=sharp_ratio)

    else:
	    return render_template('home.html',	 symbol= symbol)
	
					   

						   
