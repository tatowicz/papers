//
//  main.swift
//  cmdline
//
//  Created by Anthony on 8/14/23.
//

import Foundation
import Accelerate

print("Hello, World!")


let strideA = vDSP_Stride(1)
let strideB = vDSP_Stride(1)
let strideC = vDSP_Stride(1)


var a = [Float](repeating: .random(in: 0...1), count: 100000000)
var b = [Float](repeating: .random(in: 0...1), count: 100000000)

let n = vDSP_Length(a.count)
var c = [Float](repeating: 0, count: a.count)


let start = Date().timeIntervalSince1970
let iter = 500

// Iterate on DSP_vadd, because larger arrays take up too much memory
for _ in 1...iter {
    vDSP_vadd(a, strideA, b, strideB, &c, strideC, n)
}
    
let runtime = Date().timeIntervalSince1970 - start

print(runtime)

// Compute # of adds / seconds
let ops = Float(c.count * iter) / Float(runtime) / 1e9
print(ops, "Gops")
