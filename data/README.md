# Test material

Everything here is authored for the prerequisite spikes. None of it is real customer data.
The organisation "Кварта" and every person, number and address are fictional.

```
scripts/        call scripts, one file per call, voiced by tools/synthesize.py
cases/<id>/     one case (обращение): case.json lists its interactions
rubric/v0.json  draft rubric, the 10 questions from DESIGN-PROMPT.md
gold/<id>.json  author's expected answers for a case, written before any model run
audio/          synthesized calls: <call>.wav (8 kHz stereo, operator left,
                client right) and <call>.reference.json (exact text and timing)
```

## Call script format

```
# voice operator=marina client=filipp
# pause 0.35
O: Компания «Кварта», меня зовут Анна.
C(+1.2): Здравствуйте.          start 1.2 s after the previous utterance ends
O(~0.6): Угу.                   start 0.6 s before the previous utterance ends (overlap)
C: Номер [[четыре один семь|number]].
```

`O` is the operator, `C` the client. `[[text|type]]` marks an entity whose recognition is
scored separately; types are `number`, `sum`, `date`, `name`, `term`. Numbers are written
out as words so the reference transcript is unambiguous.

Header directives: `voice` picks Yandex SpeechKit v1 voices per role, optionally with an
emotion after a colon (`client=jane:evil`); `pause` is the default gap between turns.
