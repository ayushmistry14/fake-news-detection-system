import pandas as pd
import numpy as np
import random
from datetime import datetime

# Set seed for reproducible results
np.random.seed(42)
random.seed(42)

# ============================================================================
# POLITICS DATASET
# ============================================================================

politics_real_titles = [
    "President signs bipartisan infrastructure bill into law",
    "Senate passes major climate change legislation",
    "Congress approves new healthcare reform package",
    "Supreme Court upholds voting rights act provisions",
    "White House announces new economic stimulus plan",
    "Governors from both parties endorse education reform",
    "Federal judge blocks controversial immigration policy",
    "House passes bill to lower prescription drug prices",
    "Bipartisan group of senators introduces police reform bill",
    "President meets with foreign leaders to discuss trade agreement",
    "Unemployment rate drops to 50-year low under current administration",
    "Congress passes bill to invest $100 billion in rural broadband",
    "Supreme Court rules in favor of environmental protections",
    "White House brokers peace deal between allied nations",
    "Senate confirms new cabinet secretary in 98-2 vote",
    "Federal Reserve announces interest rate cut to boost economy",
    "Bipartisan bill to expand veterans benefits passes unanimously",
    "President hosts summit on combating domestic terrorism",
    "Congress overrides veto for first time in 5 years",
    "New ethics rules passed with overwhelming support"
]

politics_fake_titles = [
    "BREAKING: President secretly resigns - White House covering it up",
    "Congressman arrested for treason against the United States",
    "Supreme Court Justice caught accepting millions in bribes",
    "FEMA camps being built to detain political opponents",
    "Voting machines hacked to flip millions of ballots",
    "Military on standby for coup against the government",
    "Secret tape reveals candidate admitted election was rigged",
    "FBI director resigns amid corruption investigation",
    "President planning to declare martial law and cancel elections",
    "Deep state operatives arrested in midnight raid",
    "Supreme Court secretly voted to overturn landmark rights case",
    "Foreign government paying politicians to influence legislation",
    "Vaccine mandates hide microchips for government surveillance",
    "Emergency broadcast system test scheduled for election night cover-up",
    "Speaker of the House involved in international money laundering",
    "Pentagon document reveals planned false flag attack",
    "Chief Justice hospitalized after mysterious poisoning attempt",
    "White House chief of staff resigns citing criminal activity",
    "United Nations preparing to take over US sovereignty",
    "Billions in COVID relief funds funneled to offshore accounts"
]

def generate_politics_content(is_real):
    if is_real:
        return "According to official sources, the legislative body passed the measure with bipartisan support. The bill has been sent to the executive branch for final approval. Lawmakers from both sides praised the compromise. The new law will take effect in 90 days and is expected to benefit millions of Americans. Implementation will be overseen by federal agencies working with state partners."
    else:
        return "WARNING: Sources close to the administration have revealed this bombshell information that mainstream media refuses to report. Multiple insiders confirm the story, but the government is trying to suppress it. Share this before it gets censored! Independent journalists risk their lives to bring you this truth. The corrupt establishment wants you to stay asleep, but the evidence is overwhelming."

# ============================================================================
# SPORTS DATASET
# ============================================================================

sports_real_titles = [
    "Underdogs advance to championship after overtime victory",
    "Star quarterback signs record-breaking contract extension",
    "Team announces new head coach with championship pedigree",
    "Olympic gold medalist announces retirement after final season",
    "Rookie sensation breaks 30-year-old scoring record",
    "League announces expansion team coming to major city",
    "Injury update: Star player expected to return for playoffs",
    "Team trades veteran players for draft picks in rebuild move",
    "College athlete declares for professional draft",
    "World Cup champions welcomed home by thousands of fans",
    "Star pitcher throws first no-hitter in franchise history",
    "Team clinches playoff berth with dramatic final-day win",
    "Commissioner announces new safety protocols for players",
    "Hall of Fame inductee gives emotional acceptance speech",
    "Major tournament expands field to include more nations",
    "Team unveils new state-of-the-art training facility",
    "Legendary coach wins 500th career game",
    "International competition draws record television ratings",
    "Young star commits to hometown team long-term",
    "Annual charity game raises millions for local hospitals"
]

sports_fake_titles = [
    "BREAKING: Superstar demands trade after locker room brawl video leaks",
    "Referee arrested for fixing championship game outcome",
    "Team owner secretly trying to relocate franchise to Las Vegas",
    "MVP candidate failed drug test, facing season-long suspension",
    "League considering radical rule change to eliminate defense",
    "Star player faked injury to avoid playing in rival city",
    "General manager recorded making racist comments about players",
    "Team doctor suspended for prescribing banned substances",
    "World Cup host country accused of bribing selection committee",
    "Legendary announcer fired after offensive on-air rant",
    "Commissioner secretly owns betting website, investigation launched",
    "Star athlete's spouse files for divorce over gambling debts",
    "Team deliberately lost final game for better draft position",
    "Iconic stadium condemned as unsafe, will be demolished",
    "League planning to ban popular celebration after complaints",
    "Rookie's record season invalidated due to illegal equipment",
    "Coach benched star as punishment for political views",
    "International federation accused of embezzling millions",
    "Hall of Famer stripped of awards after admission of cheating",
    "Entire team quarantined after mysterious illness outbreak"
]

def generate_sports_content(is_real):
    if is_real:
        return "In an exciting matchup that came down to the final seconds, the team secured a hard-fought victory. Post-game analysis highlights the strategic adjustments made during halftime. The coach praised the players' resilience and execution. Fans erupted as the clock expired, celebrating a memorable win that will be talked about for seasons to come."
    else:
        return "EXCLUSIVE: Sources inside the locker room have confirmed the shocking truth that team officials are desperately trying to hide. A player who spoke on condition of anonymity revealed details that would rock the sports world if true. Mainstream sports media won't cover this because of advertiser pressure. Share before it's deleted!"

# ============================================================================
# TECHNOLOGY DATASET
# ============================================================================

tech_real_titles = [
    "Tech giant announces breakthrough in quantum computing",
    "New smartphone features improved camera and battery life",
    "Open-source AI model matches performance of leading competitors",
    "Major security vulnerability patched in popular software",
    "Company reports quarterly earnings exceeding analyst expectations",
    "Researchers achieve new efficiency record for solar cells",
    "Electric vehicle manufacturer opens largest factory in North America",
    "Streaming service adds 10 million subscribers in quarter",
    "Space agency successfully tests new rocket system",
    "Social media platform introduces new privacy controls",
    "Tech conference draws record attendance after return to in-person",
    "Startup raises $100 million for innovative healthcare solution",
    "New data privacy law takes effect in major jurisdiction",
    "Company announces carbon-negative commitment by 2030",
    "Popular messaging app adds end-to-end encryption by default",
    "Tech workers unionize at major corporation for first time",
    "Wireless charging standard achieves faster speeds than wired",
    "Autonomous vehicle completes cross-country journey safely",
    "New programming language gains popularity among developers",
    "Tech giant acquires gaming studio for $5 billion"
]

tech_fake_titles = [
    "Secret government AI gains consciousness, escapes laboratory",
    "Popular app caught recording users through camera without permission",
    "Tech CEO resigns after human trafficking investigation launched",
    "New iPhone feature will self-destruct if unauthorized repair attempted",
    "Major cloud provider suffers complete data loss affecting millions",
    "5G towers responsible for unexplained bird deaths worldwide",
    "Social media algorithms secretly programmed to promote depression",
    "Government demands backdoor into all encrypted messaging apps",
    "Tech company's 'carbon neutral' claim exposed as complete fraud",
    "New virus discovered that destroys smartphone batteries permanently",
    "Popular operating system contains code that spies on every keystroke",
    "Tech billionaire builds underground bunker for elite survival",
    "Major website's algorithm caught manipulating search results for profit",
    "Smart home devices hacked remotely to lock families inside homes",
    "Tech company's new update deliberately slows older devices",
    "Anonymous leaks documents proving mass surveillance program",
    "Popular fitness tracker selling health data to insurance companies",
    "Tech CEO admits products are designed to be addictive in leaked email",
    "New cryptocurrency is actually government tracking system",
    "Major tech merger blocked after antitrust investigation finds fraud"
]

def generate_tech_content(is_real):
    if is_real:
        return "The company announced the new technology at its annual developer conference, demonstrating significant improvements over previous versions. Independent testing confirmed the performance claims. Industry analysts predict widespread adoption within 18 months. The release marks an important milestone for the sector."
    else:
        return "ALERT: A trusted insider has provided documentation proving that everything you've been told is false. This technology poses a grave danger that the companies don't want you to know about. Technical experts who reviewed the evidence are sounding alarms. Share this warning with everyone you care about."

# ============================================================================
# HEALTH DATASET
# ============================================================================

health_real_titles = [
    "New study shows exercise reduces risk of heart disease by 30%",
    "FDA approves novel treatment for rare genetic disorder",
    "Researchers identify potential early warning sign for Alzheimer's",
    "Vaccination campaign credited with eliminating major disease in region",
    "Hospital system reduces readmission rates with new follow-up protocol",
    "Mediterranean diet linked to lower rates of depression, study finds",
    "Breakthrough therapy shows promise for autoimmune disease patients",
    "Sleep quality directly impacts immune function, research confirms",
    "WHO releases updated guidelines for hypertension treatment",
    "New screening method detects cancer at earlier, more treatable stage",
    "Mental health parity law improves access to therapy services",
    "Major pharmacy chain offers free health screenings nationwide",
    "Study finds meditation reduces chronic pain without medication",
    "Public health campaign reduces smoking rates to all-time low",
    "New antibiotic effective against drug-resistant bacteria",
    "Telehealth usage remains high after pandemic-era expansion",
    "Vitamin D supplementation reduces fracture risk in elderly patients",
    "Gene therapy restores hearing in children with inherited deafness",
    "Hospital-acquired infections decline following new protocols",
    "Plant-based protein sources linked to longer lifespan"
]

health_fake_titles = [
    "Government whistleblower reveals cure for cancer being suppressed",
    "Common household chemical proven to cause autism, study finds",
    "Doctors across country reporting mysterious vaccine injury syndrome",
    "Essential oil mixture cures terminal illness, pharmaceutical companies furious",
    "Health agency caught falsifying data to hide treatment dangers",
    "Chemicals in tap water causing irreversible brain damage, leaked study shows",
    "Popular weight loss supplement linked to sudden liver failure in hundreds",
    "Surgeon general resigns after admitting to covering up pandemic origin",
    "World Health Organization planning mandatory depopulation vaccines",
    "Natural herb banned because it threatens billion-dollar drug industry",
    "Flu shot contains tracking device, nurse whistleblower confesses",
    "Major study linking cell phones to brain tumors suppressed for decades",
    "Hospital billing code reveals secret euthanasia program for elderly",
    "Popular painkiller pulled from shelves after death toll reaches thousands",
    "CDC document proves masks cause carbon dioxide poisoning in children",
    "Anti-aging breakthrough discovered by retired scientist, ignored by medical establishment",
    "International health organization accused of profiting from disease",
    "Vaccinated individuals shedding harmful particles to unvaccinated population",
    "Medical journal retracts study after discovering fatal error in research",
    "Health app selling user DNA data to pharmaceutical companies"
]

def generate_health_content(is_real):
    if is_real:
        return "The peer-reviewed study, published in a major medical journal, followed 10,000 participants over five years. The findings were presented at the annual medical conference and received endorsement from leading health organizations. Experts recommend discussing these findings with healthcare providers to determine individual risk factors."
    else:
        return "CRITICAL HEALTH ALERT: Medical professionals are calling this the biggest cover-up since the tobacco industry. A courageous doctor risked their license to provide these documents. Mainstream medical organizations are part of the conspiracy. Don't wait until it's too late - natural alternatives that work exist."

# ============================================================================
# FINANCE DATASET
# ============================================================================

finance_real_titles = [
    "Federal Reserve raises interest rates by quarter point",
    "Major bank reports quarterly profits exceeding estimates",
    "Stock market closes at record high following jobs report",
    "Company announces dividend increase for 10th consecutive year",
    "Unemployment claims fall to lowest level in six months",
    "Housing market shows signs of cooling after price surge",
    "New trade agreement boosts exports between major economies",
    "Consumer spending rises 0.5% as holiday season begins",
    "Oil prices stabilize following OPEC production announcement",
    "Cryptocurrency exchange receives regulatory approval in key market",
    "Inflation rate moderates to 2.1%, meeting Fed target",
    "Merger between industry giants approved by regulators",
    "Small business confidence index reaches three-year high",
    "Manufacturing sector expands for fifth consecutive month",
    "Investor lawsuit dismissed by federal court judge",
    "Real estate investment trust announces major acquisition",
    "Credit rating agency upgrades sovereign debt outlook to stable",
    "Retail sales surge during holiday shopping season",
    "Bond yields decline amid increased demand for safe assets",
    "Financial literacy program reaches one million students"
]

finance_fake_titles = [
    "Bank insider reveals imminent collapse of major financial institution",
    "Secret currency to replace dollar next month - prepare now",
    "Government planning to seize retirement accounts under new law",
    "Investment guru predicts 90% market crash beginning tomorrow",
    "Federal Reserve secretly printing trillions causing hyperinflation",
    "Major cryptocurrency founder arrested for running Ponzi scheme",
    "Hedge fund billionaire admits market manipulation on hidden camera",
    "Credit reporting agency lost data on every American citizen",
    "Popular payment app freezing accounts without explanation",
    "Sovereign wealth fund preparing to dump US Treasury holdings",
    "Major brokerage fined billions for selling customer order flow",
    "Bankruptcy filings surge to levels not seen since 2008 crisis",
    "Insider trading ring uncovered at federal regulatory agency",
    "Counterfeit currency worth billions discovered in circulation",
    "Major corporation's financial statements completely fabricated for years",
    "Housing bubble 2.0 about to burst - expert predicted last crash",
    "Central bank secretly planning digital currency to monitor all spending",
    "Major investment bank on brink of failure, bailout imminent",
    "Pension fund underfunded by trillions, benefits will be cut",
    "Insider trading investigation expands to include dozens of executives"
]

def generate_finance_content(is_real):
    if is_real:
        return "According to the quarterly report filed with securities regulators, the company's performance reflects improving market conditions. Analysts have updated their price targets following the announcement. The CFO attributed results to cost-cutting measures and increased efficiency. Share buyback program expected to continue."
    else:
        return "FINANCIAL EMERGENCY: A senior executive has provided documents proving the system is about to collapse. Those in the know are moving their assets offshore. The warning signs are all there if you know what to look for. Don't be the last to know when it's too late to act."

# ============================================================================
# ENTERTAINMENT DATASET
# ============================================================================

entertainment_real_titles = [
    "Blockbuster sequel breaks opening weekend box office records",
    "Streaming service renews hit series for third season",
    "Academy announces new diversity requirements for Best Picture eligibility",
    "Music icon announces first tour in five years",
    "Actor joins cast of highly anticipated superhero film",
    "Film festival awards top prize to independent documentary",
    "Popular video game franchise announces new installment",
    "Emmy-nominated showrunner signs development deal with network",
    "Broadway revival receives rave reviews from critics",
    "Music streaming platform reports record revenue from subscriptions",
    "Late night host announces retirement after legendary run",
    "Comic convention draws largest attendance in history",
    "Actress wins defamation lawsuit against gossip website",
    "Former co-stars reunite for charity special event",
    "Streaming platform orders three new reality shows",
    "Movie studio announces release dates for upcoming slate",
    "Grammy winner reveals new album release date and tracklist",
    "TV network cancels struggling show after two seasons",
    "Actor's production company signs first-look deal with studio",
    "Classic film restored and re-released in theaters"
]

entertainment_fake_titles = [
    "A-list couple secretly divorced six months ago, maintaining facade",
    "Major actor fired from franchise after offensive social media posts surface",
    "Popular singer's new album entirely ghostwritten by secret team",
    "Reality TV star's relationship completely staged for ratings",
    "Oscar-winning director accused of using CGI doubles for actors",
    "Late night host's monologue jokes written by controversial AI program",
    "Movie studio paid critics for positive reviews of failed blockbuster",
    "Famous actress faked pregnancy for tabloid deal",
    "Streaming service manipulating viewership numbers to justify cancellations",
    "Music awards show voting rigged by corporate sponsors",
    "Talk show host secretly recorded making racist comments about guests",
    "Popular franchise sequel delayed indefinitely due to creative differences blowup",
    "Former child star reveals abuse by network executives in leaked memoir",
    "Reality competition show outcomes predetermined by producers",
    "Major awards show nominees chosen based on bribes, investigation finds",
    "Celebrity chef's recipes plagiarized from unknown food blogger",
    "Popular band's farewell tour actually planned for five more years",
    "Actress intentionally leaked own photos for publicity, agent confirms",
    "Game show contestant admits results were coached by producers",
    "Comedian's Netflix special jokes stolen from lesser-known comics"
]

def generate_entertainment_content(is_real):
    if is_real:
        return "The announcement was made through the star's publicist earlier today. Representatives confirmed that the project is currently in development with a planned release date next year. Fans have taken to social media to express excitement about the news. Additional details are expected to be revealed in the coming weeks."
    else:
        return "HOLLYWOOD BOMBSHELL: Multiple sources close to the situation have confirmed this story that publicists are desperately trying to kill. The truth always comes out eventually, and major media can't ignore this one. Celebs want you to believe their perfect image, but the reality is much darker."

# ============================================================================
# MAIN FUNCTION TO BUILD ALL DATASETS
# ============================================================================

def build_dataset(category_name, real_titles, fake_titles, content_generator):
    """
    Build a balanced dataset for a given category
    """
    records = []
    
    # Generate real news (250-300 records)
    num_real = random.randint(250, 300)
    for i in range(num_real):
        title = random.choice(real_titles)
        content = content_generator(is_real=True)
        records.append({
            'article_id': f"{category_name[:3].upper()}_{i+1:04d}_REAL",
            'title': title,
            'content': content,
            'category': category_name,
            'label': 'real'
        })
    
    # Generate fake news (250-300 records)
    num_fake = random.randint(250, 300)
    for i in range(num_fake):
        title = random.choice(fake_titles)
        content = content_generator(is_real=False)
        records.append({
            'article_id': f"{category_name[:3].upper()}_{i+1:04d}_FAKE",
            'title': title,
            'content': content,
            'category': category_name,
            'label': 'fake'
        })
    
    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    return df

# Build all datasets
print("Building datasets...")
print("-" * 50)

datasets = {
    'politics': build_dataset('politics', politics_real_titles, politics_fake_titles, generate_politics_content),
    'sports': build_dataset('sports', sports_real_titles, sports_fake_titles, generate_sports_content),
    'technology': build_dataset('technology', tech_real_titles, tech_fake_titles, generate_tech_content),
    'health': build_dataset('health', health_real_titles, health_fake_titles, generate_health_content),
    'finance': build_dataset('finance', finance_real_titles, finance_fake_titles, generate_finance_content),
    'entertainment': build_dataset('entertainment', entertainment_real_titles, entertainment_fake_titles, generate_entertainment_content)
}

# Save individual CSV files
for category, df in datasets.items():
    filename = f"{category}.csv"
    df.to_csv(filename, index=False)
    print(f"✓ Saved {filename}: {len(df)} records ({len(df[df['label']=='real'])} real, {len(df[df['label']=='fake'])} fake)")

# Also create separate True and Fake files for each category
for category, df in datasets.items():
    true_df = df[df['label'] == 'real'].copy()
    fake_df = df[df['label'] == 'fake'].copy()
    
    # Remove the label column for True/Fake split files
    true_df = true_df.drop(columns=['label'])
    fake_df = fake_df.drop(columns=['label'])
    
    true_df.to_csv(f"{category}_True.csv", index=False)
    fake_df.to_csv(f"{category}_Fake.csv", index=False)
    print(f"✓ Created {category}_True.csv ({len(true_df)} rows) and {category}_Fake.csv ({len(fake_df)} rows)")

print("-" * 50)
print(f"\nSummary:")
print(f"Total records across all categories: {sum(len(df) for df in datasets.values())}")

# Create a combined dataset
combined_df = pd.concat(datasets.values(), ignore_index=True)
combined_df.to_csv("complete_news_dataset.csv", index=False)
print(f"✓ Combined dataset saved as 'complete_news_dataset.csv' ({len(combined_df)} total records)")
