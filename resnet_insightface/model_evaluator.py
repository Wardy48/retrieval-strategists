def get_identity(filename):
  
    if filename.startswith('c'):
        return None  # cartoon image, no identity
    if filename.startswith('s'):
        return None  # synthetic image, no identity
    
    identity_str = filename.split('_')[0]
    return int(identity_str)

"""
def get_identity_instructor(filename):  # placeholder for dataset that will be provided during competition.
"""


def evaluate_retrieval(results, identity_func=None):

    if identity_func is None:
        identity_func = get_identity

    top_1 = 0
    top_5 = 0
    top_10 = 0
    total_queries = 0

    for query_filename, gallery_list in results.items():
        query_id = identity_func(query_filename)
        
        # skip queries with no identity
        if query_id is None:
            continue

        total_queries += 1
        gallery_ids = [identity_func(f) for f in gallery_list[:10]]

        if gallery_ids[0] == query_id:
            top_1 += 1
        
        if query_id in gallery_ids[:5]:
            top_5 += 1

        if query_id in gallery_ids[:10]:
            top_10 += 1

    if total_queries == 0:
        return {
            'top_1': 0,
            'top_5': 0,
            'top_10': 0,
            'no_of_queries': 0
        }

    return {
        'top_1': top_1 / total_queries,
        'top_5': top_5 / total_queries,
        'top_10': top_10 / total_queries,
        'no_of_queries': total_queries
    }


def score(metrics):

    #Compute the final score
  
    return (metrics['top_1'] * 600 + 
            metrics['top_5'] * 300 + 
            metrics['top_10'] * 100)


if __name__ == '__main__':
    # evaluate retrieval with fake data
    fake_results = {
        # Query 1
        '5472_2.jpg': ['5472_1.jpg', '1234_3.jpg', '8888_2.jpg', 'c_001.jpg', 's_007.jpg',
                       '9999_1.jpg', '7777_3.jpg', '6666_2.jpg', 'c_002.jpg', 's_010.jpg'],
    
    }
    
    metrics = evaluate_retrieval(fake_results)
    total_score = score(metrics)
    
    print(f"\nResults on fake test data:")
    print(f"Top-1:  {metrics['top_1']:.4f} ")
    print(f"Top-5:  {metrics['top_5']:.4f} ")
    print(f"Top-10: {metrics['top_10']:.4f}")
    print(f"Total score: {total_score:.1f} / 1000")