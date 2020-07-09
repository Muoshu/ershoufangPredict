import re

class Pipeline(object):
    def process_item(self, item, spider):
        size = item['Size']
        floor = item['Floor']
        year=item['Year']
        item['Size'] = float(re.findall(r'\d*', size)[0])
        item['Floor']= re.findall(r'\d+',floor)[0]
        item['Year'] = int(re.findall(r'\d*', year)[0])
        return item