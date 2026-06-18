"""
Fill missing Validity values in eval_lang_20260618_en_valids.csv.
Evaluations made by reading item content and judging relevance to query.
Ratings: o=best match, a=partial/acceptable, x=irrelevant
"""
import pandas as pd

CSV_PATH = "experiments/eval_lang_20260618_en_valids.csv"

# Manually evaluated: {row_index: validity}
EVALS = {
    # T3_EN: "What remains after farewell"
    125: "o",  # Older Chests (Damien Rice) - loss/farewell theme
    126: "a",  # Turning Page (Instrumental, Sleeping At Last) - new chapter after parting
    127: "o",  # End Of Me (A Day To Remember) - lyrics about aftermath of loss
    128: "x",  # A Quick One Before the Eternal Worm Devours Connecticut - dark existential, not farewell
    129: "o",  # Where Have All the Flowers Gone - classic folk farewell/loss

    # A1_EN: "A sunlit cafe"
    160: "a",  # The Trip to Italy - Italian food tour, restaurants not specifically cafe
    161: "o",  # French Roast - literally set in a Parisian café
    162: "a",  # Restaurant (1998) - restaurant setting, waiters
    163: "a",  # Bon appétit - luxury restaurant, food/romance
    164: "a",  # A Coffee in Berlin - "coffee" in title, urban drifting
    165: "o",  # Tom's Diner - "I am sitting at the diner on the corner" — perfect
    166: "a",  # On An Evening In Roma (Dean Martin) - Italian evening, Mediterranean atmosphere
    167: "o",  # Opening Up (Waitress Broadway) - set in a diner/cafe
    168: "x",  # Nice Girls Don't Stay For Breakfast - about leaving after a night, not cafe
    169: "a",  # You Matter to Me (Waitress Broadway) - romantic song in cafe context
    170: "a",  # Between Meals: An Appetite for Paris - Parisian food culture
    171: "a",  # Flavour with Benefits: France - French cuisine
    172: "o",  # Eating & Drinking in Paris: French Menu Guide - covers cafes
    173: "x",  # Milk Street Mediterranean - cooking recipes, not about cafes
    174: "o",  # The Food Lover's Guide to Paris - specifically includes cafés
    175: "a",  # Between Meals: An Appetite for Paris (all domain)
    176: "a",  # Flavour with Benefits: France (all domain)
    177: "o",  # Eating & Drinking in Paris (all domain)
    178: "x",  # Milk Street Mediterranean (all domain)
    179: "o",  # The Food Lover's Guide to Paris (all domain)

    # A2_EN: "A quiet afternoon with the sound of rain"
    180: "a",  # Al primo soffio di vento - quiet August afternoon, countryside
    181: "a",  # Les hautes solitudes - silent, contemplative film
    182: "a",  # Oto-na-ri - Japanese film, neighbors through thin walls, quiet mood
    183: "a",  # The Silence Before Bach - contemplative film about Bach's music
    184: "a",  # Une heure de tranquillité - comedy about wanting one hour of peace
    185: "o",  # Spring Rain - directly matches rain theme
    186: "a",  # Good Night Sleep - ambient sleep sounds, peaceful
    187: "o",  # Sleep Rain - specifically rain sounds
    188: "a",  # Sleeping Lotus - ambient piano, peaceful
    189: "a",  # Water Pour - water/ambient sound
    190: "a",  # Sleep Psalms - nighttime mindfulness (not afternoon but quiet/restful)
    191: "x",  # Goodnight Mind - insomnia self-help, not atmospheric
    192: "o",  # Silence: In the Age of Noise - about finding and valuing silence
    193: "a",  # The Perfume of Silence - philosophical book on silence
    194: "o",  # Silence: The Power of Quiet - about quietness and stillness
    195: "o",  # Spring Rain (all domain)
    196: "a",  # Good Night Sleep (all domain)
    197: "o",  # Sleep Rain (all domain)
    198: "a",  # Sleeping Lotus (all domain)
    199: "a",  # Water Pour (all domain)

    # A3_EN: "In front of a warm fireplace in midwinter"
    200: "a",  # Hjem til jul - Norwegian "Home for Christmas", winter homecoming
    201: "o",  # Thomas Kinkade's Christmas Cottage - cozy cottage paintings, winter warmth
    202: "a",  # Miesten vuoro - Finnish men in sauna, warmth and intimacy
    203: "o",  # Christmas in Connecticut - winter, Christmas, cozy home setting
    204: "a",  # Kansakunnan olohuone - Finnish "living room", cozy indoor space
    205: "a",  # St. Patrick's Day (John Mayer) - "break out the winter clothes", cold season
    206: "x",  # I Live My Life for You (Firehouse) - generic love song, band name not content
    207: "a",  # Near Light (Ólafur Arnalds) - ambient, wintry Icelandic atmosphere
    208: "a",  # Tomorrow's Song (Ólafur Arnalds) - ambient, contemplative
    209: "a",  # The Day Before You - Acoustic - warm acoustic tone
    210: "o",  # Cozy White Cottage Seasons - "100 Ways to Be Cozy", perfect match
    211: "a",  # Cold-Weather Cooking - cooking in cold weather, hearth-adjacent
    212: "x",  # Camping Cookbook Dutch Oven - outdoor cooking, not cozy indoor
    213: "x",  # Open Sandwiches: Smørrebrød - Nordic food recipes, not fireplace related
    214: "x",  # Cook & Ladder Company No. 1: A Firehouse Cookbook - firefighters' food, not fireplace
    215: "o",  # Cozy White Cottage Seasons (all domain)
    216: "a",  # Cold-Weather Cooking (all domain)
    217: "a",  # St. Patrick's Day (all domain)
    218: "x",  # Camping Cookbook (all domain)
    219: "x",  # Open Sandwiches (all domain)

    # A4_EN: "Empty city streets at dawn"
    220: "x",  # En la ciudad sin límites - family drama around dying father, not about streets
    221: "a",  # Metro Manila - urban Filipino city life
    222: "x",  # Los (2008) - Polish "fate" film, unclear city-street connection
    223: "a",  # Quchis dgeebi - addict loitering on Tbilisi streets
    224: "a",  # Zero Bridge - teen pickpocket in Kashmir urban streets
    225: "a",  # Beautiful Day (U2) - morning optimism, city imagery
    226: "a",  # Walk On (U2) - journey/perseverance, walking through emptiness
    227: "a",  # lovers' carvings (Bibio) - urban, atmospheric, carved names
    228: "a",  # good kid (Kendrick Lamar) - m.A.A.d city, Compton streets
    229: "x",  # Stuck In A Moment (U2) - not city/dawn related
    230: "a",  # On the Run: Fugitive Life in an American City - urban streetlife
    231: "a",  # Invisible Child: Poverty in an American City - urban NYC
    232: "a",  # No Way Home: Crisis of Homelessness - street life in cities
    233: "x",  # There Is Nothing for You Here - opportunity/memoir, not empty streets
    234: "a",  # The Other Side of Prospect - urban American city violence
    235: "x",  # En la ciudad sin límites (all domain)
    236: "a",  # Metro Manila (all domain)
    237: "a",  # On the Run (all domain)
    238: "x",  # Los (all domain)
    239: "a",  # Quchis dgeebi (all domain)

    # D1_EN: "Action movie set in space"
    240: "x",  # Make Your Own Damn Movie! - filmmaking documentary
    241: "x",  # Awesome; I Fuckin' Shot That! - Beastie Boys concert doc
    242: "x",  # Movie Movie (1978) - double-feature parody, not space
    243: "x",  # Comic-Con Episode IV: A Fan's Hope - fan convention documentary
    244: "x",  # Bring It On: In It to Win It - cheerleader comedy
    245: "x",  # Stand Out (A Goofy Movie) - children's film soundtrack
    246: "x",  # Playing with the Boys (Top Gun) - military jets, not space
    247: "x",  # Top Gun Anthem - military action, not space
    248: "x",  # Gonna Fly Now (Rocky) - sports action, not space
    249: "x",  # Once And For All (Newsies) - Broadway about newspaper boys
    250: "x",  # Action!: Professional Acting for Film and Television - acting textbook
    251: "x",  # More Popcorn Principles - cinematic storytelling guide
    252: "x",  # Running the Show: Television from the Inside - TV production
    253: "x",  # The Popcorn Principles - film storytelling guide
    254: "x",  # Directing Actors - directing textbook
    255: "x",  # Action!: Professional Acting (all domain)
    256: "x",  # More Popcorn Principles (all domain)
    257: "x",  # Running the Show (all domain)
    258: "x",  # The Popcorn Principles (all domain)
    259: "x",  # Directing Actors (all domain)

    # D2_EN: "Soulful music featuring jazz piano"
    260: "a",  # Jazz (documentary) - about jazz music broadly
    261: "a",  # Anita O'Day: The Life of a Jazz Singer - jazz documentary
    262: "a",  # The Benny Goodman Story - jazz biopic, clarinet not piano
    263: "o",  # The Eddy Duchin Story - biography of famous pianist and band-leader
    264: "a",  # Thirty Two Short Films About Glenn Gould - piano biopic, classical not jazz
    265: "a",  # Sing, Sing, Sing (Benny Goodman) - swing/jazz classic, not specifically piano
    266: "o",  # Satin Doll (Duke Ellington) - jazz piano standard
    267: "a",  # Blues My Naughty Sweetie Gives To Me - blues/jazz classic
    268: "o",  # I Hear Music (Ella Fitzgerald & Oscar Peterson) - jazz piano legend
    269: "a",  # Body and Soul (Coleman Hawkins) - jazz classic, saxophone not piano
    270: "o",  # Gentleman of Jazz: A Life in Music (Ramsey Lewis) - jazz pianist biography
    271: "o",  # Berklee Jazz Piano - jazz piano instructional book
    272: "o",  # Oscar Peterson Jazz Exercises for Piano - direct jazz piano match
    273: "a",  # Miles Davis: The Definitive Biography - jazz biography, trumpet not piano
    274: "a",  # How to Listen to Jazz - general jazz appreciation
    275: "o",  # Gentleman of Jazz (all domain)
    276: "o",  # Berklee Jazz Piano (all domain)
    277: "o",  # Oscar Peterson Jazz Exercises (all domain)
    278: "a",  # Miles Davis biography (all domain)
    279: "a",  # How to Listen to Jazz (all domain)

    # D3_EN: "Mystery thriller novel with a twist ending"
    280: "o",  # Clue (1985) - classic murder mystery with multiple endings
    281: "o",  # Prime Suspect 7: The Final Act - crime/mystery drama
    282: "o",  # Sleuth (1972) - mystery/thriller with deadly battle of wits
    283: "o",  # Miss Marple: The Moving Finger - Agatha Christie mystery
    284: "o",  # Nancy Drew: Detective (1938) - mystery about a disappearance
    285: "o",  # Discombobulate (Hans Zimmer) - Sherlock Holmes soundtrack
    286: "a",  # We Have All The Time In The World - James Bond theme, spy thriller
    287: "x",  # Take Care Of Business (Nina Simone) - jazz standard, not mystery
    288: "x",  # After Today (A Goofy Movie) - children's animated film
    289: "x",  # Puttin' on the Ritz (Fred Astaire) - classic jazz, not mystery
    290: "o",  # McNally's Caper - Mystery, Thriller & Suspense series
    291: "a",  # Tricky Twenty-Two: Stephanie Plum - mystery/comedy novel
    292: "o",  # Patricia Fisher's Mystery Adventures - Mystery, Thriller & Suspense
    293: "a",  # Off Target (A Hailey Webb Mystery) - mystery series
    294: "o",  # What Sam Knew (Patricia Fisher Mystery Adventures) - mystery
    295: "o",  # McNally's Caper (all domain)
    296: "a",  # Tricky Twenty-Two (all domain)
    297: "o",  # Patricia Fisher's Mystery Adventures (all domain)
    298: "a",  # Off Target (all domain)
    299: "o",  # What Sam Knew (all domain)

    # D4_EN: "A film about a romance between two people"
    300: "x",  # Double Play: James Benning and Richard Linklater - documentary about directors
    301: "o",  # The Comedian (2012) - Drama/Romance in London
    302: "a",  # Deux de la Vague - relationship between Truffaut/Godard, not romantic film
    303: "o",  # Sulemani Keeda - Comedy/Romance where romance gets in the way
    304: "o",  # The Short & Curlies - short comedy about romance between a woman and a man
    305: "x",  # Keep It Gay (The Producers) - Broadway number about show aesthetics
    306: "a",  # We Had Today - intimate instrumental, relationship theme
    307: "o",  # Glasgow Love Theme from "Love Actually" - directly from a romance film
    308: "a",  # I Could Write a Book - romantic standard
    309: "a",  # I Could Write a Book (duplicate entry)
    310: "o",  # Writing The Romantic Comedy - about crafting funny love stories for screen
    311: "x",  # The Coen Brothers book - general film history
    312: "x",  # Film Studies: An Introduction - general film studies
    313: "x",  # Leonard Maltin's Movie Guide - general movie reference
    314: "o",  # The Way We Were: Making of a Romantic Classic - about a famous romance film
    315: "o",  # Writing The Romantic Comedy (all domain)
    316: "x",  # Double Play (all domain)
    317: "o",  # The Comedian (all domain)
    318: "a",  # Deux de la Vague (all domain)
    319: "o",  # Sulemani Keeda (all domain)
}


def main():
    df = pd.read_csv(CSV_PATH)

    # Normalize existing O/A/X → o/a/x
    df["Validity"] = df["Validity"].apply(
        lambda v: v.lower() if isinstance(v, str) and v.strip() else v
    )

    # Fill empty rows
    filled = 0
    for idx, val in EVALS.items():
        if idx in df.index:
            df.at[idx, "Validity"] = val
            filled += 1

    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"Filled {filled} rows. Remaining empty: {df['Validity'].isna().sum()}")
    print("Validity distribution:")
    print(df["Validity"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
