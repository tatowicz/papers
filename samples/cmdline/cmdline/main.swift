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

let iter = 500

let start = Date().timeIntervalSince1970
// Iterate on DSP_vopp because larger arrays take up too much memory
for _ in 1...iter {
    vDSP_vmul(a, strideA, b, strideB, &c, strideC, n)
}
let runtime = Date().timeIntervalSince1970 - start

print(runtime)

// Compute # of muls / seconds
let ops = Float(c.count * iter) / Float(runtime) / 1e9
print(ops, "Gops")


// FFT on accelerate
var real = [Float](repeating: .random(in: 0...1), count: 10000)
var imaginary = [Float](repeating: 0.0, count: real.count)


let length = vDSP_Length(floor(log2(Float(real.count))))
let weights = vDSP_create_fftsetup

