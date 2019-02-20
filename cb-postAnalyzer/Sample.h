/*
 *  Copyright 2014-2017 Hui Zhang
 *  All rights reserved.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef SAMPLE_H
#define SAMPLE_H 

#include <vector>
#include <string>
#include <iostream>
#include <unordered_map>

struct SourceLocator
{
  unsigned int ID;
  std::string fileName;
  int line;
};

//this includes the device/global functions
struct GpuFunction 
{
  unsigned int funcID;
  unsigned int contextID;
  unsigned int moduleID;
  unsigned int functionIndex; //unique symbol index for this function in the module
  std::string name; //linkage name
  std::string realName; //reserved for de-mangled name, retrieved from the static analysis
};

struct Sample
{
  unsigned int srcID;
  unsigned int corrID;
  unsigned int funcID;
  //unsigned long long pcOffset;
  unsigned int sampleIdx;
  //keep the #occurance of the same sample
  unsigned int occurance;
};

struct kernel
{
  //reserved for gpu kernels
};

struct samp_key_t
{ 
  unsigned int srcID;
  unsigned int corrID;
  unsigned int funcID;
  samp_key_t() : srcID(-1),corrID(-1),funcID(-1) {}
  samp_key_t(unsigned int sid, unsigned int cid, unsigned int fid)
  : srcID(sid),corrID(cid),funcID(fid) {}
};
 
//hash function using samp_key_t as a key type
struct key_hash : public std::unary_function<samp_key_t, std::size_t> 
{
  std::size_t operator()(const samp_key_t &sk) const {
    return sk.srcID ^ sk.corrID ^ sk.funcID;
  }
};

struct key_equal : public std::binary_function<samp_key_t, samp_key_t, bool>
{
  bool operator()(const samp_key_t &sk1, const samp_key_t &sk2) const {
    return (sk1.srcID == sk2.srcID && sk1.corrID == sk2.corrID
            && sk1.funcID == sk2.funcID);
  }
};

typedef std::unordered_map<unsigned int, SourceLocator> sourceMap;
typedef std::unordered_map<unsigned int, GpuFunction> functionMap;
typedef std::vector<Sample> sampleVec;
typedef std::unordered_map<const samp_key_t, Sample, key_hash, key_equal> sampleMap;

#endif
