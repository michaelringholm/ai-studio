import os
import requests
import json
import locale
import hashlib
import urllib.parse
import om_logging as oml

def download_json(url):
    response = requests.get(url)
    response.encoding = 'UTF-8'
    data = response.json()    
    return data

def download_json_gzip(url):
    response = requests.get(url)
    response.encoding = 'utf-8'

    if response.headers.get('Content-Encoding') == 'gzip':
        import gzip
        import io
        decompressed_data = gzip.GzipFile(fileobj=io.BytesIO(response.content))
        data = json.load(decompressed_data)
    else:
        data = response.json()
    return data
    #print(json.dumps(data, indent=4, ensure_ascii=False))


def printHouseDetails():
    print("hh")

def formatMoney(amount):    
    locale.setlocale(locale.LC_ALL, 'da_DK.utf8') 
    return locale.currency(amount, grouping=True)

def url_encode(text):
    return urllib.parse.quote_plus(text)

def searchHouses(municipalities:str, cities:str=None, roads:str=None, zipCodes:int=None, addressType:str="villa", limit:int=10):
    addressTypes="villa,condo,terraced house,holiday house,farm,hobby farm" # overwrite for now    
    # municipalities=Greve
    # cities=Tune
    entriesPerPage=100 # No more than 200, then it will fail
    currentPage=1
    pagesNeeded=1 if limit / entriesPerPage < 1 else limit / entriesPerPage
    houses=None
    if(limit<entriesPerPage): entriesPerPage=limit
    try:
        while(currentPage<=pagesNeeded):
            url=f"https://api.boligsiden.dk/search/addresses?per_page={entriesPerPage}&page={currentPage}&registrationTypes=normal&addressTypes={url_encode(addressTypes)}"
            if(municipalities!=None): url+=f"&municipalities={municipalities}"
            if(cities!=None): url+=f"&cities={url_encode(cities)}"
            if(roads!=None): url+=f"&roads={roads}"
            if(zipCodes!=None): url+=f"&zipCodes={zipCodes}"
            #print(url)
            if(isCached(url)==True):
                page_result=readFromCache(url)
                oml.debug(f"loaded page from cache {currentPage}")
            else:
                page_result=download_json(url)
                cacheSearch(url,page_result)
                oml.debug(f"downloaded page {currentPage}")
            currentPage+=1        
            page_addresses=page_result['addresses']
            if(page_addresses==None): break # No more houses
            if(houses==None): houses=page_result
            else: houses['addresses'].extend(page_addresses)
    except Exception as e:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        oml.error(f"url was={url}")
        oml.error(f"url has was={url_hash}")
        oml.error(e)
        raise e
    return houses

def readFromCache(searchUrl):
    url_hash = hashlib.md5(searchUrl.encode()).hexdigest()
    file_name=f"../data/housing/{url_hash}.json"
    with open(file_name, 'r', encoding='utf-8') as f:
        data=json.load(f)
    return data

def cacheSearch(searchUrl:str,data:str):
    url_hash = hashlib.md5(searchUrl.encode()).hexdigest()
    file_name=f"../data/housing/{url_hash}.json"
    if not os.path.exists(file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def isCached(searchUrl:str):
    url_hash = hashlib.md5(searchUrl.encode()).hexdigest()
    file_name=f"../data/housing/{url_hash}.json"
    return os.path.exists(file_name)

def printHouseDetails(address):
    oml.success(f"{address['road']['name']} {address['houseNumber']}, {address['zip']['zipCode']} {address['zip']['name']}")
    oml.info(f" City={address['city']['name']}")
    if('buildings' in address):
        buildings=address['buildings']
        for building in buildings:
            oml.info(f" TotalArea={building['totalArea']} | BuildingName={building['buildingName']}")
            if('heatingInstallation' in building): oml.info(f" Heating={building['heatingInstallation']}")
            if('roofingMaterial' in building): oml.info(f" Roofing={building['roofingMaterial']}")
            if('kitchenCondition' in building): oml.info(f" Kitchen={building['kitchenCondition']}")
            if('bathroomCondition' in building): oml.info(f" Bathroom={building['bathroomCondition']}")
            if('numberOfBathrooms' in building): oml.info(f" NumberOfBathrooms={building['numberOfBathrooms']}")
            if('numberOfFloors' in building): oml.info(f" Floors={building['numberOfFloors']}")
            if('numberOfRooms' in building): oml.info(f" Rooms={building['numberOfRooms']}")
            if('totalArea' in building): oml.info(f" TotalArea={building['totalArea']}")
            if('yearBuilt' in building): oml.info(f" YearBuilt={building['yearBuilt']}")
    else: oml.warn("No buildings found for this address!")
    registrations=address['registrations']        
    for registration in registrations:
        oml.info(f" SalesPrice={formatMoney(registration['amount'])} | SalesDate={registration['date']}")
    return 