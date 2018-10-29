from __future__ import print_function


def il_scrape(prp, ncmp=1):
    """
    webscraper tool for accessing backend data in the ILThermo
    browser. A few property codes are listed below, additional
    property codes can usually be found by opening the inspection
    console in your web brwoser and searching for "dijit dijitReset
    dijitInline dijitLeft dijitDownArrowButton dijitSelect
    dijitValidationTextBox" and looking at the input value in the html

    Paremeters
    ----------
    prp : str
        property ID. Current options are:
        thermal conductivity: VvTh
        electrical conductivity: bqdZ
        viscosity: blXM
        melting temp: lcRG
        heat capacity: bvSs
    ncmp : int
        number of components in IL. Default 1

    Returns
    -------
    data : json structure
    """


    paper_url = "http://ilthermo.boulder.nist.gov/ILT2/ilsearch?" \
                "cmp=&ncmp={}&year=&auth=&keyw=&prp={}".format(ncmp, prp)

    r = requests.get(paper_url)
    papers = r.json()['res']
    i = 1
    data_url = 'http://ilthermo.boulder.nist.gov/ILT2/ilset?set={paper_id}'
    data = []
    for paper in papers[:]:
        r = requests.get(data_url.format(paper_id=paper[0]))
        data.append(r)
        sleep(0.5)  # import step to avoid getting banned by server
        i += 1
    return data