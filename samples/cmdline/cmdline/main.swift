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

print("Generating data...")

var a = (1...100000000).map( {_ in Float.random(in: 0...1)} )
var b = (1...100000000).map( {_ in Float.random(in: 0...1)} )

print("Done")

let n = vDSP_Length(a.count)
var c = [Float](repeating: 0, count: a.count)


// Just make bigger arrays, can rip through GBs in miliseconds
// Otherwise whats the point, realized looping might take
// away some of the data parallelism
let size = Float(MemoryLayout.size(ofValue: a)) * Float(a.count) * 2 / 1e9
print(size, "GB")
print(a.count, "MULs")

var start = Date().timeIntervalSince1970
vDSP_vmul(a, strideA, b, strideB, &c, strideC, n)
var runtime = Date().timeIntervalSince1970 - start

print(runtime, "seconds")

// Compute # of muls / seconds
// Cold starts are slower, warm starts can 2x or 3x performace
let ops = Float(a.count) / Float(runtime) / 1.0e9
print(ops, "Gops")

// Classic multiply
print("Classic MUL...")
start = Date().timeIntervalSince1970
for i in 0..<a.count {
    c[i] = a[i] * b[i]
}
runtime = Date().timeIntervalSince1970 - start

print(runtime, "seconds")


// FFT on accelerate
var real = [Float](repeating: .random(in: 0...1), count: 10000)
var imaginary = [Float](repeating: 0.0, count: real.count)


let length = vDSP_Length(floor(log2(Float(real.count))))
let weights = vDSP_create_fftsetup


print("Done")
