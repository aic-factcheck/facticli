from __future__ import annotations

from .constraints import FACT_CHECK_DOMAINS, url_domain
from .contracts import SourceEvidence, SourceTier

# Deterministic authority tiering. LLM judges weight sources by how often they
# are mentioned rather than by authority (AuthorityBench, 2026); exposing an
# explicit tier next to each source in the judge input is the cheap fix. The
# table is intentionally conservative: anything unknown is ``other``.

PRIMARY_SUFFIXES: tuple[str, ...] = (
    ".gov",
    ".mil",
    ".edu",
    ".int",
    ".gov.uk",
    ".gov.au",
    ".gov.cz",
    ".gov.sk",
    ".gov.pl",
    ".gc.ca",
    ".ac.uk",
    ".ac.at",
    ".edu.au",
    ".europa.eu",
    ".gouv.fr",
    ".bund.de",
    ".admin.ch",
    ".go.jp",
    ".go.kr",
    ".gob.es",
    ".gov.it",
    ".gov.in",
    ".gov.br",
    ".gov.za",
    ".gov.ie",
    ".gov.pt",
    ".gov.gr",
    ".gov.hu",
    ".gov.ro",
    ".gov.si",
    ".gov.hr",
    ".gov.lv",
    ".gov.lt",
    ".gov.ee",
    ".gov.il",
    ".gov.sg",
    ".gov.nz",
    ".govt.nz",
    ".gv.at",
    ".overheid.nl",
    ".rijksoverheid.nl",
)

PRIMARY_DOMAINS: frozenset[str] = frozenset(
    {
        # statistics offices and central banks
        "czso.cz",
        "statistics.sk",
        "stat.gov.pl",
        "destatis.de",
        "ons.gov.uk",
        "bls.gov",
        "census.gov",
        "bea.gov",
        "cdc.gov",
        "nih.gov",
        "fda.gov",
        "eurostat.ec.europa.eu",
        "ec.europa.eu",
        "ecb.europa.eu",
        "federalreserve.gov",
        "cnb.cz",
        "nbs.sk",
        "nbp.pl",
        "bankofengland.co.uk",
        "bundesbank.de",
        "imf.org",
        "worldbank.org",
        "data.worldbank.org",
        "oecd.org",
        "data.oecd.org",
        "who.int",
        "un.org",
        "data.un.org",
        "unesco.org",
        "unicef.org",
        "wto.org",
        "nato.int",
        "iea.org",
        "ipcc.ch",
        "esa.int",
        "nasa.gov",
        "noaa.gov",
        # legislatures, courts, official records
        "congress.gov",
        "senate.gov",
        "house.gov",
        "whitehouse.gov",
        "supremecourt.gov",
        "uscourts.gov",
        "federalregister.gov",
        "govinfo.gov",
        "parliament.uk",
        "legislation.gov.uk",
        "psp.cz",
        "senat.cz",
        "vlada.cz",
        "hrad.cz",
        "nrsr.sk",
        "sejm.gov.pl",
        "senat.gov.pl",
        "bundestag.de",
        "bundesregierung.de",
        "europarl.europa.eu",
        "consilium.europa.eu",
        "curia.europa.eu",
        "echr.coe.int",
        "icj-cij.org",
        "icc-cpi.int",
        "sec.gov",
        "justice.gov",
        "fbi.gov",
        "state.gov",
        "defense.gov",
        "treasury.gov",
        "irs.gov",
        "elections.cz",
        "volby.cz",
        "fec.gov",
        # scientific publishing and registries
        "nature.com",
        "science.org",
        "sciencemag.org",
        "cell.com",
        "thelancet.com",
        "nejm.org",
        "bmj.com",
        "jamanetwork.com",
        "pnas.org",
        "plos.org",
        "journals.plos.org",
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "doi.org",
        "dx.doi.org",
        "sciencedirect.com",
        "link.springer.com",
        "onlinelibrary.wiley.com",
        "academic.oup.com",
        "cambridge.org",
        "ieeexplore.ieee.org",
        "dl.acm.org",
        "aclanthology.org",
        "clinicaltrials.gov",
        "cochranelibrary.com",
        "arxiv.org",
    }
)

REFERENCE_DOMAINS: frozenset[str] = frozenset(
    {
        "wikipedia.org",
        "wikidata.org",
        "wikisource.org",
        "britannica.com",
        "archive.org",
        "web.archive.org",
        "archive.ph",
        "archive.today",
        "ourworldindata.org",
        "statista.com",
        "macrotrends.net",
        "tradingeconomics.com",
        "data.gov",
        "loc.gov",
        "europeana.eu",
        "jstor.org",
        "scholar.google.com",
        "semanticscholar.org",
        "openalex.org",
        "crossref.org",
        "orcid.org",
        "encyclopedia.com",
        "merriam-webster.com",
        "oxfordreference.com",
        "dictionary.cambridge.org",
        "imdb.com",
        "boxofficemojo.com",
        "the-numbers.com",
        "olympics.com",
        "fifa.com",
        "uefa.com",
        "worldathletics.org",
        "guinnessworldrecords.com",
        "nobelprize.org",
        "ballotpedia.org",
        "opensecrets.org",
        "followthemoney.org",
        "govtrack.us",
        "c-span.org",
        "hansard.parliament.uk",
    }
)

NEWS_DOMAINS: frozenset[str] = frozenset(
    {
        # wires
        "reuters.com",
        "apnews.com",
        "afp.com",
        "bloomberg.com",
        "upi.com",
        "ctk.cz",
        "tasr.sk",
        "pap.pl",
        "dpa.com",
        "ansa.it",
        "efe.com",
        "kyodonews.net",
        # anglophone
        "bbc.com",
        "bbc.co.uk",
        "nytimes.com",
        "washingtonpost.com",
        "wsj.com",
        "ft.com",
        "theguardian.com",
        "economist.com",
        "cnn.com",
        "nbcnews.com",
        "cbsnews.com",
        "abcnews.go.com",
        "npr.org",
        "pbs.org",
        "politico.com",
        "politico.eu",
        "axios.com",
        "thehill.com",
        "propublica.org",
        "usatoday.com",
        "latimes.com",
        "chicagotribune.com",
        "bostonglobe.com",
        "time.com",
        "theatlantic.com",
        "newyorker.com",
        "vox.com",
        "cnbc.com",
        "marketwatch.com",
        "telegraph.co.uk",
        "thetimes.co.uk",
        "independent.co.uk",
        "news.sky.com",
        "channel4.com",
        "itv.com",
        "irishtimes.com",
        "rte.ie",
        "cbc.ca",
        "theglobeandmail.com",
        "globalnews.ca",
        "abc.net.au",
        "smh.com.au",
        "theage.com.au",
        "nzherald.co.nz",
        "stuff.co.nz",
        "scmp.com",
        "japantimes.co.jp",
        "nhk.or.jp",
        "aljazeera.com",
        "timesofindia.indiatimes.com",
        "thehindu.com",
        "indianexpress.com",
        "haaretz.com",
        "timesofisrael.com",
        # continental europe
        "dw.com",
        "spiegel.de",
        "zeit.de",
        "faz.net",
        "sueddeutsche.de",
        "tagesschau.de",
        "welt.de",
        "handelsblatt.com",
        "orf.at",
        "derstandard.at",
        "nzz.ch",
        "srf.ch",
        "swissinfo.ch",
        "lemonde.fr",
        "lefigaro.fr",
        "liberation.fr",
        "france24.com",
        "francetvinfo.fr",
        "rfi.fr",
        "elpais.com",
        "elmundo.es",
        "rtve.es",
        "corriere.it",
        "repubblica.it",
        "rainews.it",
        "nos.nl",
        "nrc.nl",
        "volkskrant.nl",
        "vrt.be",
        "lesoir.be",
        "svt.se",
        "dn.se",
        "svd.se",
        "dr.dk",
        "politiken.dk",
        "nrk.no",
        "aftenposten.no",
        "yle.fi",
        "hs.fi",
        "euronews.com",
        "euobserver.com",
        # czech, slovak, polish
        "irozhlas.cz",
        "ceskatelevize.cz",
        "ct24.ceskatelevize.cz",
        "idnes.cz",
        "novinky.cz",
        "seznamzpravy.cz",
        "denikn.cz",
        "aktualne.cz",
        "hn.cz",
        "ihned.cz",
        "lidovky.cz",
        "e15.cz",
        "respekt.cz",
        "echo24.cz",
        "denik.cz",
        "blesk.cz",
        "reflex.cz",
        "sme.sk",
        "dennikn.sk",
        "aktuality.sk",
        "pravda.sk",
        "hnonline.sk",
        "tvnoviny.sk",
        "ta3.com",
        "rtvs.sk",
        "stvr.sk",
        "wyborcza.pl",
        "rp.pl",
        "onet.pl",
        "wp.pl",
        "tvn24.pl",
        "polsatnews.pl",
        "tvp.info",
        "polskieradio.pl",
        "money.pl",
        "bankier.pl",
        "oko.press",
    }
)

USER_GENERATED_DOMAINS: frozenset[str] = frozenset(
    {
        "twitter.com",
        "x.com",
        "t.co",
        "facebook.com",
        "fb.com",
        "instagram.com",
        "threads.net",
        "tiktok.com",
        "youtube.com",
        "youtu.be",
        "reddit.com",
        "quora.com",
        "medium.com",
        "substack.com",
        "wordpress.com",
        "blogspot.com",
        "blogger.com",
        "tumblr.com",
        "linkedin.com",
        "pinterest.com",
        "t.me",
        "telegram.me",
        "vk.com",
        "rumble.com",
        "bitchute.com",
        "odysee.com",
        "truthsocial.com",
        "gab.com",
        "parler.com",
        "gettr.com",
        "mastodon.social",
        "bsky.app",
        "discord.com",
        "fandom.com",
        "wikia.com",
        "answers.com",
        "answers.yahoo.com",
        "stackexchange.com",
        "stackoverflow.com",
        "4chan.org",
        "4channel.org",
        "9gag.com",
        "imgur.com",
        "flickr.com",
        "twitch.tv",
        "patreon.com",
        "change.org",
        "gofundme.com",
        "scribd.com",
        "slideshare.net",
        "prezi.com",
        "wattpad.com",
        "goodreads.com",
        "yelp.com",
        "tripadvisor.com",
        "trustpilot.com",
        "glassdoor.com",
        "indeed.com",
        "meetup.com",
    }
)


def _host_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def _matches_any(host: str, domains: frozenset[str]) -> bool:
    return any(_host_matches(host, domain) for domain in domains)


def _fact_check_host(host: str, path: str) -> bool:
    host_and_path = host + path
    for entry in FACT_CHECK_DOMAINS:
        if "/" in entry:
            if host_and_path.startswith(entry):
                return True
        elif _host_matches(host, entry):
            return True
    return False


def classify_source_tier(url: str, extra_tiers: dict[str, SourceTier] | None = None) -> SourceTier:
    """Return the authority tier for a URL.

    Precedence: caller-provided overrides, fact-checker table, user-generated
    platforms, primary domains and suffixes, reference works, news outlets,
    then ``other``. Fact-checkers are classified before primary domains so
    that a fact-check section hosted on a news or government domain keeps
    its fact-checker tier.
    """
    host = url_domain(url)
    if not host:
        return SourceTier.OTHER
    try:
        from urllib.parse import urlparse

        path = urlparse(url).path.lower()
    except ValueError:
        path = ""

    if extra_tiers:
        for domain, tier in extra_tiers.items():
            if _host_matches(host, domain.lower().strip().removeprefix("www.")):
                return tier

    if _fact_check_host(host, path):
        return SourceTier.FACT_CHECKER
    if _matches_any(host, USER_GENERATED_DOMAINS):
        return SourceTier.USER_GENERATED
    if _matches_any(host, PRIMARY_DOMAINS):
        return SourceTier.PRIMARY
    if any(host.endswith(suffix) for suffix in PRIMARY_SUFFIXES):
        return SourceTier.PRIMARY
    if _matches_any(host, REFERENCE_DOMAINS):
        return SourceTier.REFERENCE
    if _matches_any(host, NEWS_DOMAINS):
        return SourceTier.NEWS
    return SourceTier.OTHER


def assign_source_tiers(
    sources: list[SourceEvidence],
    extra_tiers: dict[str, SourceTier] | None = None,
) -> list[SourceEvidence]:
    """Return copies of the sources with ``tier`` filled in where missing."""
    tiered: list[SourceEvidence] = []
    for source in sources:
        if source.tier is not None:
            tiered.append(source)
            continue
        tiered.append(source.model_copy(update={"tier": classify_source_tier(source.url, extra_tiers)}))
    return tiered


def tier_counts(sources: list[SourceEvidence]) -> dict[str, int]:
    """Count sources per tier for summaries and judge context."""
    counts: dict[str, int] = {}
    for source in sources:
        key = source.tier.value if source.tier is not None else "untiered"
        counts[key] = counts.get(key, 0) + 1
    return counts
