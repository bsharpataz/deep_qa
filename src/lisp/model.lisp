(define word-family (word-parameters entity-parameters)
  (lambda (word)
    (lambda (entity)
      (if (dictionary-contains entity entities)
        (let ((var (make-entity-var entity)))
          (make-inner-product-classifier
            var #t (get-cat-word-params word word-parameters) (get-entity-params entity entity-parameters))
          var)
        #f)
      )))

(define word-rel-family (word-rel-params entity-tuple-params)
  (define word-rel (word)
    (lambda (entity1 entity2)
      (if (dictionary-contains (list entity1 entity2) entity-tuples)
        (let ((var (make-entity-var (cons entity1 entity2))))
          (make-inner-product-classifier
            var #t (get-rel-word-params word word-rel-params)
            (get-entity-tuple-params entity1 entity2 entity-tuple-params))
          var)
        #f
        )
      ))
  word-rel)

(define expression-family (parameters)
  (let ((word-parameters (get-ith-parameter parameters 0))
        (entity-parameters (get-ith-parameter parameters 1))
        (word-rel-parameters (get-ith-parameter parameters 2))
        (entity-tuple-parameters (get-ith-parameter parameters 3))
        (word-cat (word-family word-parameters entity-parameters))
        (word-rel (word-rel-family word-rel-parameters entity-tuple-parameters)))
    (define expression-evaluator (expression entities)
      (eval expression))
    expression-evaluator))

(define word-ranking-family (word-parameters entity-parameters)
  (lambda (word)
    (lambda (entity neg-entity)
      (if (dictionary-contains entity entities)
        (let ((var (make-entity-var entity)))
          (make-ranking-inner-product-classifier
            var #t (get-cat-word-params word word-parameters) (get-entity-params entity entity-parameters)
            (get-entity-params neg-entity entity-parameters))
          var)
        #f)
      )))

(define word-rel-ranking-family (word-rel-params entity-tuple-params)
  (define word-rel (word)
    (lambda (entity1 neg-entity1 entity2 neg-entity2)
      (if (dictionary-contains (list entity1 entity2) entity-tuples)
        (let ((var (make-entity-var (cons entity1 entity2))))
          (make-ranking-inner-product-classifier
            var #t (get-rel-word-params word word-rel-params)
            (get-entity-tuple-params entity1 entity2 entity-tuple-params)
            (get-entity-tuple-params neg-entity1 neg-entity2 entity-tuple-params))
          var)
        #f
        )
      ))
  word-rel)

(define expression-ranking-family (parameters)
  (let ((word-parameters (get-ith-parameter parameters 0))
        (entity-parameters (get-ith-parameter parameters 1))
        (word-rel-parameters (get-ith-parameter parameters 2))
        (entity-tuple-parameters (get-ith-parameter parameters 3))
        (word-cat (word-ranking-family word-parameters entity-parameters))
        (word-rel (word-rel-ranking-family word-rel-parameters entity-tuple-parameters)))
    (eval define-expression-evaluator)
    expression-evaluator))

(define expression-parameters
  (make-parameter-list (list (make-parameter-list (array-map (lambda (x) (make-vector-parameters latent-dimensionality))
                                                             (dictionary-to-array cat-words)))
                             (make-parameter-list (array-map (lambda (x) (make-vector-parameters latent-dimensionality))
                                                             (dictionary-to-array entities)))
                             (make-parameter-list (array-map (lambda (x) (make-vector-parameters latent-dimensionality))
                                                             (dictionary-to-array rel-words)))
                             (make-parameter-list (array-map (lambda (x) (make-vector-parameters latent-dimensionality))
                                                             (dictionary-to-array entity-tuples)))
                             )))

(define print-parameters (parameters)
  (let ((word-parameters (get-ith-parameter parameters 0))
        (entity-parameters (get-ith-parameter parameters 1))
        (word-rel-parameters (get-ith-parameter parameters 2))
        (entity-tuple-parameters (get-ith-parameter parameters 3)))
    ; (array-map (lambda (word) (display word (parameters-to-string (get-cat-word-params word word-parameters))))
    ;           (dictionary-to-array cat-words))

    (array-map (lambda (word) (display word (parameters-to-string (get-rel-word-params word word-rel-parameters))))
               (dictionary-to-array rel-words))
    ))

