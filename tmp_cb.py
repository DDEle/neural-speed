from transformers import AutoTokenizer
from neural_speed import Model
import multiprocessing as mp
import os
import sys
from typing import List

model_name = "/home/wangzhe/dingyi/models/finetuned-gptj"
# model_bin = '/home/wangzhe/dingyi/models/finetuned-gptj-pr83-q4-j-int8-pc.bin'
model_bin = '/home/wangzhe/dingyi/models/3bit-gpt-j-6b-gptq-int8.bin'


prompts = [
    "Summarize the following news article: (CNN)Share, and your gift will be multiplied. That may sound like an esoteric adage, but when Zully Broussard selflessly decided to give one of her kidneys to a stranger, her generosity paired up with big data. It resulted in six patients receiving transplants. That surprised and wowed her. \"I thought I was going to help this one person who I don't know, but the fact that so many people can have a life extension, that's pretty big,\" Broussard told CNN affiliate KGO. She may feel guided in her generosity by a higher power. \"Thanks for all the support and prayers,\" a comment on a Facebook page in her name read. \"I know this entire journey is much bigger than all of us. I also know I'm just the messenger.\" CNN cannot verify the authenticity of the page. But the power that multiplied Broussard's gift was data processing of genetic profiles from donor-recipient pairs. It works on a simple swapping principle but takes it to a much higher level, according to California Pacific Medical Center in San Francisco. So high, that it is taking five surgeons, a covey of physician assistants, nurses and anesthesiologists, and more than 40 support staff to perform surgeries on 12 people. They are extracting six kidneys from donors and implanting them into six recipients. \"The ages of the donors and recipients range from 26 to 70 and include three parent and child pairs, one sibling pair and one brother and sister-in-law pair,\" the medical center said in a statement. The chain of surgeries is to be wrapped up Friday. In late March, the medical center is planning to hold a reception for all 12 patients. Here's how the super swap works, according to California Pacific Medical Center. Say, your brother needs a kidney to save his life, or at least get off of dialysis, and you're willing to give him one of yours. But then it turns out that your kidney is not a match for him, and it's certain his body would reject it. Your brother can then get on a years-long waiting list for a kidney coming from an organ donor who died. Maybe that will work out -- or not, and time could run out for him. Alternatively, you and your brother could look for another recipient-living donor couple like yourselves -- say, two more siblings, where the donor's kidney isn't suited for his sister, the recipient. But maybe your kidney is a match for his sister, and his kidney is a match for your brother. So, you'd do a swap. That's called a paired donation. It's a bit of a surgical square dance, where four people cross over partners temporarily and everybody goes home smiling. But instead of a square dance, Broussard's generous move set off a chain reaction, like dominoes falling. Her kidney, which was removed Thursday, went to a recipient, who was paired with a donor. That donor's kidney went to the next recipient, who was also paired with a donor, and so on. On Friday, the last donor will give a kidney to someone who has been biding time on one of those deceased donor lists to complete the chain. Such long-chain transplanting is rare. It's been done before, California Pacific Medical Center said in a statement, but matching up the people in the chain has been laborious and taken a long time. That changed when a computer programmer named David Jacobs received a kidney transplant. He had been waiting on a deceased donor list, when a live donor came along -- someone nice enough to give away a kidney to a stranger. Jacobs paid it forward with his programming skills, creating MatchGrid, a program that genetically matches up donor pairs or chains quickly. \"When we did a five-way swap a few years ago, which was one of the largest, it took about three to four months. We did this in about three weeks,\" Jacobs said. But this chain wouldn't have worked so quickly without Broussard's generosity -- or may not have worked at all. \"The significance of the altruistic donor is that it opens up possibilities for pairing compatible donors and recipients,\" said Dr. Steven Katznelson. \"Where there had been only three or four options, with the inclusion of the altruistic donor, we had 140 options to consider for matching donors and recipients.\" And that's divine, Broussard's friend Shirley Williams wrote in a comment her on Broussard's Facebook page. \"You are a true angel my friend.\"",
    "Summarize the following news article: (CNN)On the 6th of April 1996, San Jose Clash and DC United strode out in front of 31,683 expectant fans at the Spartan Stadium in San Jose, California. The historic occasion was the first ever Major League Soccer match -- a brave new dawn for the world's favorite sport in a land its charms had yet to conquer. Summarizing the action for ESPN, commentator Ty Keough eagerly described the momentous \"birth of a new era for American soccer.\" Looking back at footage from that balmy evening now it's hard not to feel a certain nostalgia. Baggy shirts, questionable hairstyles and strange rule adaptations to make games more exciting were all part of the formative MLS experience. Countdown clocks were employed to provide drama at the end of each half. Even more bizarrely, tied games were settled by shootouts that saw attacking players run with the ball from 35-yards out before attempting to beat the opposing goalkeeper. As the MLS prepares to mark the beginning of its 20th season, it's hard to comprehend just how much the league has progressed in the intervening period. Long gone is the desire to tamper with the rules of the game for a start. Attendances are higher than ever before while the number of teams involved has doubled from 10 in the 1996 campaign to 20 in 2015. A further four are set to be added by 2020. On top of this, the new season is the first of a new domestic TV and media rights deal with FOX, ESPN and Univision worth $700 million over eight years. This figure may pale beside the $5.1 billion recently paid by UK broadcasters for the English Premier League, the richest football league in the world, but it represents a tripling in value of the previous MLS deal. According to Phil Rawlins, co-primary owner and president of the new MLS franchise, Orlando City Soccer Club, \"the industry and the game itself has moved on dramatically\" in the U.S.. He believes what would equal 50 years growth in most other industries has been experienced in the first two decades of the MLS. Rawlins' club is a prime example of this rapid transformation. He describes players being pushed out of changing facilities because of a schedule clash with a yoga class not so long ago. This weekend 60,000 fans are expected to witness Orlando City's opening weekend fixture against New York City, another new club making their MLS bow. World Cup winners Kaka and David Villa will turn out for Orlando and New York City respectively. \"We're just on the crest of the wave at the moment,\" Rawlins said of football's American progress. \"Can it be the number two, number three sport in this country? Yes, I think it can. And it can be in a short space of time.\" These positive assertions are backed by the huge interest U.S. fans showed in last year's World Cup in Brazil. Team USA's group stage clash with Portugal attracted 25 million viewers, according to figures from TV ratings firm, Nielsen. That's considerably more than the 15 million baseball's 2013 World Series averaged on FOX or the similar audience that tuned into the 2014 NBA finals on ABC. Anyone who saw 20,000 pumped-up young fans pack out Chicago's Grant Park to cheer on their country via big screens, meanwhile, would find it hard to argue against soccer in the U.S. now being anything other than a big deal. Reaching this promising stage, however, has been anything but a smooth ride. The MLS was reported to have lost as much as $250 million in its first five years while average attendances initially dwindled after the inaugural season. Three teams -- Miami Fusion, Tampa Bay Mutiny (both in 2001) and Chivas USA (2014) -- were disbanded along the way due to a mixture of lack of fan interest and ownership troubles. A report by Forbes at the end of 2013, meanwhile, claimed that only 10 out of 19 MLS teams were profitable. And as recently as this week, MLS players looked like they could be going on strike over wages and the right of players to become free agents when their contracts end. Then there's the way the league develops, attracts and trades players. A salary cap restricts the amount teams can spend on playing squads. Each side, however, has a number of spaces that can be allocated to \"off budget\" signings which are not included within the cap. This includes promising Generation Adidas players who enter the MLS through the draft systems before completing their college education. Homegrown players from club's development academies are also exempt as are a maximum of three designated players (DPs), usually stellar international names whose wages and transfer fees will be covered by club owners or sponsors. One of the main criticisms of the MLS and its complex player acquisition rulebook is that while it does entice prominent stars of the game like David Beckham, Freddie Ljungberg and Thierry Henry to appear in the MLS, it only does so when their careers are on a downward trajectory. Why would an exceptional player want to move to a league that can only attract a handful of top talents at any one time, after all? And herein lies one of the leagues biggest challenges in attracting and keeping the talented players fans want to see. Although the likes of the salary cap encourages fiscal probity, it means MLS teams are restricted by rules clubs in other markets are not. Head coach of Sporting Kansas, Peter Vermes, highlighted these difficulties in comments carried by the Kansas City Star newspaper last year. \"We're in a place where at times you can't compete with foreign clubs because of the kind of dynamics they have in regards to finances. We have a salary cap. They don't,\" Vermes said. According to Paulo Teixeira, a football agent who has worked to bring in and sell players from the league in recent years, current philosophies with regards player-trading may be have to be tweaked to help the MLS grow yet further. He describes the importance of placing an emphasis on attracting younger players with European passports. Such talented individuals will have a sell-on value that can be recouped by the league and their clubs if they move on from the MLS to the biggest and wealthiest leagues across the Atlantic. Theoretically, at least, this money can then be reinvested in the league, player development and attracting yet more promising players to the MLS. This in turn will raise the standard further. An early example of this strategy can perhaps be found in the transfer of Oriol Rossell, a Spanish midfielder who moved from Sporting Kansas to Sporting Lisbon last year in a deal brokered by Teixeira. Rossell arrived on a free transfer aged 20 after being released by FC Barcelona in 2012. He excelled at Kansas, winning the MLS Cup before being sold to the Portuguese giants at a profit in June 2014. Teixeira is quick to make clear such plans would need good scouting systems to truly flourish. It could also be achieved by signing DPs closer to the peak stage of their career, he added. This last point is something that appears be happening already. \"Before they used to have a lot of big names who could no longer run in Europe,\" Teixeira said. \"(But) Villa is not an old guy, (Frank) Lampard is still going strong\" and both could still offer something to teams in Europe, he said by way of example of New York City's first DP signings. Nevertheless, he continued, the signing of more young players with big potential  \"is probably something we'll see more of.\" Whether Teixeira is correct will become apparent in the months and years ahead. Either way, that brave new MLS dawn that broke over San Jose back in 1996 has turned into a bright morning. CNN's Don Riddell contributed to this story.",
    "Summarize the following news article: (CNN)French striker Bafetimbi Gomis, who has a history of fainting, said he is now \"feeling well\" after collapsing during Swansea's 3-2 loss at Tottenham in the Premier League on Wednesday. The worrying incident occurred in the first half at White Hart Lane -- after Tottenham scored in the seventh minute -- but the 29-year-old left the pitch conscious following about five minutes of treatment. The Guardian added that he was wearing an oxygen mask. Play was temporarily stopped before resuming. As the match progressed, Swansea tweeted that Gomis was \"fine,\" with manager Garry Monk using the same word to describe Gomis' condition. Gomis spent the night in hospital as a precaution, Swansea said on its website. \"I wanted to reassure you concerning my health,\" Gomis told the website. \"It actually looks much scarier than it is physically dangerous, and I am feeling well now. \"I have been under a great deal of stress and fatigue due to my father's health, which requires me to go back and forth from France. \"I was disappointed that I couldn't help my team tonight, but now everything is back in order. I also want to thank everyone for their support and get well messages.\" Gomis had similar fainting spells in France, which prompted the president of his former club, Jean-Michel Aulas of Lyon, to tell French television in 2009: \"We can't not be worried, it scares you each time.\" Swansea ran tests on Gomis, said Monk, prior to signing him on a free transfer last July. \"He just has a little bit of low blood pressure which causes you a little bit of problems,\" Monk said in a televised interview on Sky. \"It's been part of his life. We were well aware of that when we signed him. He's done all the hospital checks and all the medical checks you can possibly do and it's just part of his life. \"It's no problems whatsoever. It's not as serious as it looks.\" Gomis has scored two league goals for Swansea this season, mostly in a backup role. He became the Welsh side's top striker when Wilfried Bony signed with Manchester City in January. Almost exactly three years ago at White Hart Lane, then Bolton midfielder Fabrice Muamba collapsed after suffering a cardiac arrest. He was near death,  according to Bolton, but survived after being treated at the London Chest Hospital. He subsequently retired. Other footballers, including Cameroon international Marc-Vivien Foe in 2003 and Spanish international Antonio Puerta in 2007, didn't survive after collapsing on the pitch.",
    "Summarize the following news article: (CNN)French striker Bafetimbi Gomis, who has a history of fainting, said he is now \"feeling well\" after collapsing during Swansea's 3-2 loss at Tottenham in the Premier League on Wednesday. The worrying incident occurred in the first half at White Hart Lane -- after Tottenham scored in the seventh minute -- but the 29-year-old left the pitch conscious following about five minutes of treatment. The Guardian added that he was wearing an oxygen mask. Play was temporarily stopped before resuming. As the match progressed, Swansea tweeted that Gomis was \"fine,\" with manager Garry Monk using the same word to describe Gomis' condition. Gomis spent the night in hospital as a precaution, Swansea said on its website. \"I wanted to reassure you concerning my health,\" Gomis told the website. \"It actually looks much scarier than it is physically dangerous, and I am feeling well now. \"I have been under a great deal of stress and fatigue due to my father's health, which requires me to go back and forth from France. \"I was disappointed that I couldn't help my team tonight, but now everything is back in order. I also want to thank everyone for their support and get well messages.\" Gomis had similar fainting spells in France, which prompted the president of his former club, Jean-Michel Aulas of Lyon, to tell French television in 2009: \"We can't not be worried, it scares you each time.\" Swansea ran tests on Gomis, said Monk, prior to signing him on a free transfer last July. \"He just has a little bit of low blood pressure which causes you a little bit of problems,\" Monk said in a televised interview on Sky. \"It's been part of his life. We were well aware of that when we signed him. He's done all the hospital checks and all the medical checks you can possibly do and it's just part of his life. \"It's no problems whatsoever. It's not as serious as it looks.\" Gomis has scored two league goals for Swansea this season, mostly in a backup role. He became the Welsh side's top striker when Wilfried Bony signed with Manchester City in January. Almost exactly three years ago at White Hart Lane, then Bolton midfielder Fabrice Muamba collapsed after suffering a cardiac arrest. He was near death,  according to Bolton, but survived after being treated at the London Chest Hospital. He subsequently retired. Other footballers, including Cameroon international Marc-Vivien Foe in 2003 and Spanish international Antonio Puerta in 2007, didn't survive after collapsing on the pitch.",
]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token  # gpt-j has no pad_token
pad_token = tokenizer(tokenizer.pad_token)['input_ids'][0]
# inputs = tokenizer(prompts, padding=True, return_tensors='pt').input_ids

inputs = tokenizer(prompts, padding=False, return_tensors=None).input_ids

SEED = 1234
N_CTX = 2048
TOP_K = 40
TOP_P = 1.
REPETITION_PENALTY = 1.5
LENGTH_PENALTY = 1.
TEMPERATURE = .9
BEAM_SEARCH = True
N_PREDICT = 128
BEAM_SIZE = 4
MIN_NEW_TOKENS = 30
DO_EARLY_STOPPING = True
BATCH_SIZE = 12

NUM_SOCKET = 2
CPU_OFFSET = 0

CPUS_PER_SOCKET = 56
WORKS_PER_SOCKET = 1
CPUS_PER_WORKER = CPUS_PER_SOCKET // WORKS_PER_SOCKET

OUT_THREAD_IDX = 0

# inputs = ([inputs[1][:1024]] * 10)[:BATCH_SIZE]
inputs = (inputs * 10)[:BATCH_SIZE]
print([len(xs) for xs in inputs])

queue = mp.Queue()


def handler(i: int) -> List[int]:
    os.sched_setaffinity(0, range(CPU_OFFSET+i*CPUS_PER_WORKER, CPU_OFFSET+(i+1)*CPUS_PER_WORKER))
    if i != OUT_THREAD_IDX:
        null_fh = open(os.devnull, 'w')
        os.dup2(null_fh.fileno(), sys.stdout.fileno())
        os.dup2(null_fh.fileno(), sys.stderr.fileno())
    # if i != 0:
    #     return
    model = Model()
    model.model_type = 'gptj'
    model.init_from_bin(
        model.model_type,
        model_bin,
        max_new_tokens=N_PREDICT,
        n_batch=N_CTX,
        ctx_size=N_CTX,
        seed=SEED,
        threads=CPUS_PER_WORKER,
        repetition_penalty=REPETITION_PENALTY,
        num_beams=BEAM_SIZE,
        do_sample=False,
        top_k=TOP_K,
        top_p=TOP_P,
        temperature=TEMPERATURE,
        min_new_tokens=MIN_NEW_TOKENS,
        length_penalty=LENGTH_PENALTY,
        early_stopping=DO_EARLY_STOPPING,
        n_keep=0,
        n_discard=-1,
        shift_roped_k=False,
        batch_size=BATCH_SIZE,
        pad_token=-1,
        memory_dtype="auto",
        continuous_batching=True,
        max_request_num=BATCH_SIZE,
        model_scratch_enlarge_scale=1,
    )

    # inputs = tokenizer(prompts, padding=True, return_tensors='pt').input_ids
    # outputs = model.generate(inputs, num_beams=4, max_new_tokens=10, min_new_tokens=10, early_stopping=True,
    #                              pad_token=pad_token, continuous_batching=True, max_request_num=4, memory_dtype='f16')
    # pass pad_token if set it

    out = model.model.generate(inputs)
    print("out", out)
    queue.put((i, out))
    model.model.print_time()
    if i != OUT_THREAD_IDX:
        null_fh.close()

    # ans = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # for a in ans:
    #     print(a)
    # print("===========================")


# works = [mp.Process(target=handler, args=(i, )) for i in range(WORKS_PER_SOCKET*NUM_SOCKET)]
# [w.start() for w in works]
# [w.join() for w in works]
handler(0)

outputs = [None for _ in range(WORKS_PER_SOCKET*NUM_SOCKET)]
while not queue.empty():
    i, o = queue.get()
    outputs[i] = o

print([len(xs) for xs in inputs])

for i_inst, inst_o in enumerate(outputs):
    if inst_o is None:
        continue
    if i_inst != OUT_THREAD_IDX:
        continue
    for i_sample, o in enumerate(inst_o):
        print('==========', i_inst, i_sample, '==========',)
        print(tokenizer.decode(o))
