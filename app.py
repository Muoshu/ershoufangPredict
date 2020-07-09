import urllib.parse

from flask import Flask, render_template, request

from analysis.Analysis import price_info, get_predict, price_distribute, size_price, layout_price, renovation_price, \
    district_price, recommend

app = Flask(__name__, static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/price_info", methods=['GET'])
def get_price_info():
    city = request.values.get('city')
    region = urllib.parse.unquote(request.values.get('region'))
    c = price_info(city, region)
    return c.dump_options_with_quotes()


@app.route("/price_distribute", methods=['GET'])
def get_price_distribute():
    city = request.values.get('city')
    region = urllib.parse.unquote(request.values.get('region'))
    c = price_distribute(city, region)
    return c.dump_options_with_quotes()


@app.route("/district_price", methods=['GET'])
def get_district_price():
    city = request.values.get('city')
    region = urllib.parse.unquote(request.values.get('region'))
    c = district_price(city, region)
    return c.dump_options_with_quotes()


@app.route("/size_price", methods=['GET'])
def get_size_price():
    city = request.values.get('city')
    region = urllib.parse.unquote(request.values.get('region'))
    c = size_price(city, region)
    return c.dump_options_with_quotes()


@app.route('/layout_price', methods=['GET'])
def get_layout_price():
    city = request.values.get('city')
    region = urllib.parse.unquote(request.values.get('region'))
    c = layout_price(city, region)
    return c.dump_options_with_quotes()


@app.route('/renovation_price', methods=['GET'])
def get_renovation_price():
    city = request.values.get('city')
    region = urllib.parse.unquote(request.values.get('region'))
    c = renovation_price(city, region)
    return c.dump_options_with_quotes()


@app.route('/price_estimate', methods=['GET'])
def get_price_estimate():
    # 参数准备
    city = request.values.get('city')
    region = urllib.parse.unquote(request.values.get('region'))
    district = int(request.values.get('district'))
    size = int(request.values.get('size'))
    layout = int(request.values.get('layout'))
    type = int(request.values.get('type'))

    if district == -1:
        district = 1
    if size == -1:
        size = 70
    if layout == -1:
        layout = 1
    if type == -1:
        type = 1

    info = [city, region, district, size, type, layout]
    c = get_predict(info)
    return c.dump_options_with_quotes()


@app.route('/recommend', methods=['GET'])
def get_recommend():
    city = request.values.get('city')
    region = urllib.parse.unquote(request.values.get('region'))
    district = urllib.parse.unquote(request.values.get('district'))
    size = int(request.values.get('size'))
    layout = urllib.parse.unquote(request.values.get('layout'))
    info = [city, region, district, layout, size]
    c = recommend(info)
    return c.dump_options_with_quotes()


if __name__ == "__main__":
    app.run()
