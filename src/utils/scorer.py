def highest_score(lst, func):
    if len(lst) == 0:
        return None
    scores = []
    for o in lst:
        scores.append(func(o))
    # for i in range(len(lst)):
    #     if lst[i]["lang"] not in ["English", "Big 5 code", "Chinese BG code"]:
    #         continue
    #     log.info(f"{scores[i]}. {lst[i]}")
    max_score = max(scores)
    if max_score < 0:
        return None
    return lst[scores.index(max_score)]
