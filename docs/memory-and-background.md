# Memory, notes and background jobs

Jig keeps what it knows about you on this computer, in its own database. You can read it, change it and delete it. This is a beta (version 0.1.0b2). Memory search is a word index (SQLite FTS5). Embeddings are not in this version.

The rules for what is sent to a cloud model, and how deletion is done on disk, are in the README under [Safety model](../README.md#safety-model) (the part headed "Memory and notes") and [Using a cloud model](../README.md#using-a-cloud-model). This page is how you use them.

## Memory and notes

Open **Settings > Memory and notes**.

1. **Memories** are what Jig remembers about you. Search with **Search**, or **Show all**. Open one to view it, edit it or delete it.
2. **Tell Jig something to remember** adds one yourself. **Kind** defaults to `fact`. **Tags** are comma-separated.
3. You can also tell Jig in the chat to remember something. Changing or forgetting a memory, when the agent does it because you asked, is reviewed by the Sentinel and may need your OK. Jig does not rewrite your memories by itself.
4. **Jig's notes** are what it writes down for itself while it works, such as research findings and reminders. They are separate from memories.
5. **Delete all notes…** removes every note. Memories stay.
6. **Forget everything…** deletes every memory and note, every conversation, and every finished job with its results. Jig then starts again knowing nothing about you. A conversation Jig is replying in, or a job that is still going, has to finish or be stopped first. Those are left, and Jig says how many it kept.

With a cloud model, memories that were already sent in a request stay with the provider under its terms. Deleting them in Jig removes the copy on this computer.

## Keyed facts, skills, and what past runs did

Alongside the flat memories, Jig keeps three keyed stores in the same database, with the same rules: nothing leaves the computer, and **Forget everything…** deletes them all.

1. **Keyed facts** are what Jig has learned, filed by *scope* (the author, one book, one project) and *kind*: a `preference` of yours, a fact of the `world` of a book, the `continuity` of the work in progress, or an `open_question` a failed job left behind. The agent files them as it works; an exact repeat is stored once, and a correction marks the old row `superseded` instead of deleting it. Facts Jig recorded are shown to it at the end of each message (newest first, or the ones that best match the message with `memory_prompt = "relevant"`), bounded to 1,500 characters, beside the memories' own 6,000, and each line says which scope it came from. Which *work* a conversation is about is said once and kept: name it under the message box in the page, with `jig chat --scope book:<slug>`, or with `"scope"` in the `POST /chat` body, and every later turn of that conversation is given that work's facts and the author's only, so a chapter of one book never arrives with another book's canon. Say nothing and Jig reads one off your message - but only when the message points at exactly one work; a message that could belong to two is answered with every scope rather than guessed at, and the page says which it chose. Under the message box the page always says which of these happened ("About book:ember-island, as this conversation said", "Jig could not tell which work your message meant, so it was given every scope"), and you can always widen it back: **Every work** (or `all`) says deliberately every scope, and clearing the box (or `auto`) puts the conversation back to reading one per turn. A wrongly narrowed conversation hides facts, so what Jig answered in is written under every message rather than left to guess. Facts you wrote yourself are never rewritten by Jig, and Jig never rewrites your memories by itself - that promise is unchanged.
2. **Skills** are procedures the agent writes for itself after a task finally worked: candidates at first, and a candidate changes nothing and is never shown to the model. You make one active only by your own decision (the `PATCH /skills/{id}` API route sets the status; a Settings control is to come). Active skills appear in each new conversation as a short index of names and one-line summaries (bounded to 1,200 characters, 12 entries); Jig reads a full body only when it decides it needs one. A skill body describes how to produce something; it is checked at write time so it cannot just quote passages of the work itself.
3. **The trajectory** is a bounded index of what past runs did: which tools ran, what succeeded, what failed and with what error type, and how tasks ended. It never holds what was said - no message, no tool argument, no result, no error text, exactly like the audit log - and entries cannot be edited. Jig can read it (`history_search`), so a job resembling one that went wrong before can find that out for itself; you can too, at `GET /trajectory`.

**Forget everything…** now also deletes every fact, every skill (at any status) and the whole trajectory index, with the same secure deletion as everything else. The audit log still records only that it happened, with counts, never content.

## Conversations and jobs

**Settings > Conversations and jobs** lists conversations and finished jobs. You can read one and delete one, or delete them all. Deleting a conversation or a finished job removes the copies of its text Jig kept (the conversation, the runs, the steps, and the approval details). Files Jig wrote in its folder stay: those are yours.

**Settings > History** is the audit log. Entries cannot be edited or deleted, so they do not hold what was said. They record what happened, when, and what was decided.

## Schedules (background jobs)

A schedule is a job Jig runs by itself at set times. Each run is an ordinary task, so the usual rules, the Sentinel and approvals apply. The full repeat rules are in the README under [Schedules](../README.md#schedules).

1. Open **Settings > Schedules** and **Add a schedule**.
2. Give it a **Name** and **What should Jig do each time?**
3. Choose **Repeats**: every weekday (Monday to Friday), every day, on days you choose, every few minutes or hours, or a cron expression. For a clock time, set **At** (the form starts at 08:00).
4. Choose **While it runs**:
   - **Just look, don't touch** is the default for a new schedule. Jig can read and take notes. It will not change, send or save anything of yours.
   - **Can act (asks when needed)** may take actions, still subject to the Sentinel and your approval.
5. Choose **Add schedule**. The list shows the next run and how the last one ended.

Pause or resume a schedule from that list, or delete it. **Timezone** on the same page is Jig's timezone for schedules. The default is the computer's timezone. If you change it, schedules that were on the old timezone keep their times of day.

A one-off background job is a goal. Open **What Jig's up to**, describe what Jig should achieve, and choose **Create goal**. Jig plans the steps, works through them in the background, and asks you before anything that needs your OK. **Stop** there stops background jobs after you confirm.

While Jig is paused (the **Pause** button beside **What Jig's up to**), schedules do not start new work.

Closing Jig's window on Windows does not turn Jig off, so schedules still run. `jig stop`, or **Turn Jig off**, does stop them until Jig is running again.
