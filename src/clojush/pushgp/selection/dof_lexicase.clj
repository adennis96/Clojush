(ns clojush.pushgp.selection.dof-lexicase
  (:use [clojush matrix random]
        [clojure.math numeric-tower]))

(defn diff-cost
  [A B]
  (reduce + (flatten (matr-map (fn [x y] (expt (- x y) 2)) A B))))

(defn nmf
  "Given non-negative matrix V, returns two matrices W and H such that V = WH
  Implemented using multiplicative update rule (Lee and Seung, 2001)"
  [V k max-iter]
  (loop [W (rand-matr (height V) k 1)
         H (rand-matr k (width V) 1)
         iter max-iter]
    (println iter (diff-cost V (matr-mult W H))) ; DEBUG
    (if (or (<= iter 0) (= (diff-cost V (matr-mult W H)) 0))
      [W H]
      (let [new-H (matr-doall (matr-map * H
                    (matr-map / (matr-mult (transpose W) V)
                                (matr-mult (transpose W) W H))))]
        (recur
          (matr-doall (matr-map * W
            (matr-map / (matr-mult V (transpose new-H))
                        (matr-mult W new-H (transpose new-H)))))
          new-H
          (- iter 1))))))

(defn assign-features-to-individual
  [individual features]
  (assoc individual :dof-features features))

(defn calculate-dof-features
  "Calculates feature vectors for each individual in the population based
  on matrix factorization"
  [pop-agents {:keys [use-single-thread dof-features dof-iterations] :as argmap}]
  (println "calculating DOF features...") (flush)
  (let [error-matrix (map (fn [x] (:errors (deref x))) pop-agents)
        [W H] (nmf error-matrix dof-features dof-iterations)]
    ; (println "V-WH: " (matr-map - error-matrix (matr-map round (matr-mult W H)))) ; DEBUG
    (dorun (map (fn [indiv feats] ((if use-single-thread swap! send)
                                   indiv assign-features-to-individual feats))
                pop-agents W)))
  (when-not use-single-thread (apply await pop-agents))
  (println "done calculating DOF features"))

(defn dof-lexicase-selection
  "Returns the individual that performs the best on the dof-features when
  considered one at a time in a random order"
  [pop argmap]
  (loop [survivors pop
         cases (lshuffle (range (count (:dof-features (first pop)))))]
    (if (or (empty? cases)
            (empty? (rest survivors))
            (< (lrand) (:lexicase-slippage argmap)))
      (lrand-nth survivors)
      (let [min-feat (apply min (map (fn [feats] (nth feats (first cases)))
                                     (map :dof-features survivors)))]
        (recur (filter (fn [indiv] (= (nth (:dof-features indiv) (first cases))
                                      min-feat))
                       survivors)
               (rest cases))))))
