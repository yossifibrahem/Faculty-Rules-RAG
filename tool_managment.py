from SearchRules import search_rules

def RAG(query, top_k=1):
    rules_results = search_rules(query, db_name="rules", top_k=top_k)
    RAG = rules_results
    return RAG

def FAQ(query, top_k=1):
    FAQ_results = search_rules(query, db_name="FAQ", top_k=top_k)
    return FAQ_results

def links():
    course_content = {
       "This is a Drive with all courses content" : "https://drive.google.com/drive/folders/18IqwQ9pL0i2H4sqwoywP6_Hd_wz7_BKu",
       "the official website of the university" : "https://alexu.edu.eg/index.php/en/",
       "the official website of the college" : "https://fcds.alexu.edu.eg/index.php/en/",
       "the stuednt's registration site" : "https://gs.alexu.edu.eg/FCDS/index.php",
    }
    return course_content
