import os
import requests
import time
from tqdm import tqdm


class CompanyGenerator:

    def __init__(self):
        self.kb_url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
        self.company_query = """
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
            SELECT ?company ?label
            WHERE {
                ?company wdt:P31/wdt:P279* wd:Q4830453;
                         rdfs:label ?label.
                FILTER(LANG(?label) = "en").
            }
            """
        self.aliases_query = """
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?altLabel 
            WHERE {
                wd:$$company$$ skos:altLabel ?altLabel
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            """
        self.companies = None
        self.company_aliases = None

    def download(self, use_cache=True):
        self.companies = self.get_companies(use_cache)
        self.company_aliases = self.get_company_aliases(use_cache)

    def query_kb(self, url, query):
        return requests.get(url, params={'query': query, 'format': 'json'}).json()

    def get_companies(self, use_cache):
        companies = {}
        if use_cache and os.path.exists('../data/companies.tsv'):
            print('read companies cache')
            with open('../data/companies.tsv') as f:
                for line in f.readlines():
                    items = line.split('\t')
                    company_id = items[0]
                    company_label = items[1]
                    companies[company_id] = company_label
        else:
            print('download companies')
            with open('../data/companies.tsv', 'w') as f:
                data = self.query_kb(self.kb_url, self.company_query)
                for result in data["results"]["bindings"]:
                    company_id = result['company']['value'].split('/')[-1]
                    company_label = result['label']['value']
                    companies[company_id] = company_label
                    f.write(company_id + '\t' + company_label + '\n')
        return companies

    def get_company_aliases(self, use_cache):
        company_aliases = {}
        if use_cache and os.path.exists('../data/company_aliases.tsv'):
            print('read company aliases cache')
            with open('../data/company_aliases.tsv', 'r') as f:
                for line in f.readlines():
                    items = line.strip().split('\t')
                    if len(items) >= 2:
                        company = items[0]
                        company_label = items[1]
                        company_aliases[company] = (company_label, items[2:])
        are_they_all = True
        for company in self.companies:
            if company not in company_aliases:
                are_they_all = False
                break
        if not are_they_all:
            print('download company aliases')
            with open('../data/company_aliases.tsv', 'a+') as f:
                for n, company in tqdm(enumerate(self.companies), total=len(self.companies)):
                    if company not in company_aliases:
                        # print(n, company, self.companies[company])
                        query = self.aliases_query.replace("$$company$$", company)
                        try:
                            data = self.query_kb(self.kb_url, query)
                            aliases = []
                            for result in data["results"]["bindings"]:
                                aliases.append(result['altLabel']['value'])

                            company_aliases[company] = (self.companies[company], aliases)
                            f.write(company + '\t' + self.companies[company] + '\t' + '\t'.join(aliases) + '\n')
                            time.sleep(1)
                        except:
                            print('exception, wait for 2s')
                            time.sleep(2)
        return company_aliases


if __name__ == '__main__':
    CompanyGenerator().download()
