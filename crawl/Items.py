import scrapy
class Item(scrapy.Item):
    # define the fields for your item here like:
    Id = scrapy.Field()
    Region = scrapy.Field()
    Garden = scrapy.Field()
    Layout = scrapy.Field()
    Size = scrapy.Field()
    Direction = scrapy.Field()
    Renovation = scrapy.Field()
    Type = scrapy.Field()
    Floor = scrapy.Field()
    Year = scrapy.Field()
    Price = scrapy.Field()
    District = scrapy.Field()
    pass