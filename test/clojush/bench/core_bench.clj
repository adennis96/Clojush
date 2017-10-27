(ns clojush.bench.core-bench
  (:require [libra.bench :refer :all]

            [clojush.core :refer [-main]]
            [clojush.test.core-test :refer [reset-globals!]]))

(defn call-main [args]
  (reset-globals!)
  (with-out-str (apply -main args)))

(defmacro defbench-main [name args]
  `(defbench ~(symbol (str "main-" name))
    (with-redefs [shutdown-agents (fn [])]
      (is (dur 10 (call-main ~args))))))

(def configurations
  {:jan-13
    ["clojush.problems.software.replace-space-with-newline"
      ":autoconstructive" "true"
      ":autoconstructive-genome-instructions" ":uniform"
      ":autoconstructive-diversification-test" ":size-and-instruction"
      ":autoconstructive-si-children" "2"
      ":autoconstructive-integer-rand-enrichment" "10"
      ":autoconstructive-boolean-rand-enrichment" "10"
      ":max-points" "1600"
      ":final-report-simplifications" "0"
      ":report-simplifications" "0"
      ":max-genome-size-in-initial-program" "400"
      ":evalpush-limit" "1600"
      ":parent-selection" ":leaky-lexicase"]
   :nth-prime
    ["clojush.problems.integer-regression.nth-prime"
     ":final-report-simplifications" "0"
     ":report-simplifications" "0"]})

(defbench-main "autocon-10-gen-jan-13"
  (concat (:jan-13 configurations) [":max-generations" "10"]))

(defbench-main "autocon-10-gen-jan-13-serial"
  (concat (:jan-13 configurations) [":max-generations" "10" ":use-single-thread" "true"]))


(defbench-main "nth-prime-10-gen"
  (concat (:nth-prime configurations) [":max-generations" "10"]))

(defbench-main "nth-prime-10-gen-serial"
  (concat (:nth-prime configurations) [":max-generations" "10" ":use-single-thread" "true"]))


(defbench cleanup
  (shutdown-agents))
