(ns clojush.pushgp.selection.dof-lexicase
  (:use [clojush random globals]
        [clojush.pushgp.selection epsilon-lexicase]
        [clojure.math numeric-tower]
        [uncomplicate.neanderthal core native]
        [uncomplicate.fluokitten core]))

(defn rand-matr
  [h w n]
  (dge h w (for [x (range h) y (range w)] (rand n))))

(defn diff-cost
  [A B]
  (fold (fmap (fn ^double [^double x ^double y] (expt (- x y) 2)) A B)))

(defn nmf
  "Given non-negative matrix V, returns two matrices W and H such that V = WH
  Implemented using multiplicative update rule (Lee and Seung, 2001)"
  [V k max-iter]
  (loop [W (rand-matr (mrows V) k 1)
         H (rand-matr k (ncols V) 1)
         iter max-iter
         start-time (System/currentTimeMillis)]
    (printf "step %2d; cost: %.4f\n" iter (diff-cost V (mm W H))) (flush) ; DEBUG
    (if (or (<= iter 0) (= (diff-cost V (mm W H)) 0))
      [W H]
      (let [new-H (fmap (fn ^double [^double x ^double y] (* x y)) H
                        (fmap (fn ^double [^double x ^double y] (/ x y))
                              (mm (trans W) V)
                              (mm (trans W) W H)))
            new-W (fmap (fn ^double [^double x ^double y] (* x y)) W
                        (fmap (fn ^double [^double x ^double y] (/ x y))
                              (mm V (trans new-H))
                              (mm W new-H (trans new-H))))]
        (printf "time: %.4f sec; " (float (/ (- (System/currentTimeMillis) start-time) 1000))) ; DEBUG
        (recur
          new-W
          new-H
          (- iter 1)
          (System/currentTimeMillis))))))

(defn assign-features-to-individual
  [individual features]
  (assoc individual :dof-features features))

(defn calculate-dof-features
  "Calculates feature vectors for each individual in the population based
  on matrix factorization"
  [pop-agents {:keys [use-single-thread dof-features dof-iterations] :as argmap}]
  (println "calculating DOF features...") (flush)
  (let [error-vecs (map (fn [x] (:errors (deref x))) pop-agents)
        error-matrix (dge (count error-vecs) (count (first error-vecs)) (flatten error-vecs) {:layout :row})
        [W H] (nmf error-matrix dof-features dof-iterations)]
    (dorun (map (fn [indiv feats] ((if use-single-thread swap! send)
                                   indiv assign-features-to-individual (into [] feats)))
                pop-agents (rows W))))
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

(defn calculate-epsilons-for-dof-epsilon-lexicase
 "Calculates the epsilon values for DOF epsilon lexicase selection. Only runs
 once per generation. "
 [pop-agents {:keys [epsilon-lexicase-epsilon]}]
 (when (not epsilon-lexicase-epsilon)
   (let [pop (map deref pop-agents)
         features (apply map list (map :dof-features pop))
         epsilons (map mad features)]
     (println "Epsilons for DOF epsilon lexicase:" epsilons)
     (reset! epsilons-for-epsilon-lexicase epsilons))))

(defn dof-epsilon-lexicase-selection
  "Returns an individual that performs within epsilon of the best on the
  dof-features when considered one at a time in random order"
  [pop {:keys [epsilon-lexicase-epsilon epsilon-lexicase-probability] :as argmap}]
  (loop [survivors pop
         cases (lshuffle (range (count (:dof-features (first pop)))))]
    (if (or (empty? cases)
            (empty? (rest survivors))
            (< (lrand) (:lexicase-slippage argmap)))
      (lrand-nth survivors)
      (let [; If epsilon-lexicase-epsilon is set in the argmap, use it for epsilon.
             ; Otherwise, use automatic epsilon selections, which are calculated once per generation.
             epsilon (if (<= (lrand) epsilon-lexicase-probability)
                       (if epsilon-lexicase-epsilon
                         epsilon-lexicase-epsilon
                         (nth @epsilons-for-epsilon-lexicase (first cases)))
                       0)
             min-feat-for-case (apply min (map #(nth % (first cases))
                                              (map :dof-features survivors)))]
        (recur (filter #(<= (nth (:dof-features %)
                                 (first cases))
                            (+ min-feat-for-case
                               epsilon))
                       survivors)
               (rest cases))))))
