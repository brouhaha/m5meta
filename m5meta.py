#!/usr/bin/python3

# m5meta: Meta assembler
# Copyright 2019 Eric Smith <spacewar@gmail.com>
# SPDX-License-Identifier: GPL-3.0

# This program is free software: you can redistribute it and/or modify
# it under the terms of version 3 of the GNU General Public License
# as published by the Free Software Foundation.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__version__ = '1.0.4'
__author__ = 'Eric Smith <spacewar@gmail.com>'

__all__ = ['__version__', '__author__',
           'M5Meta', 'M5MetaError']

import argparse
from collections import Counter, OrderedDict
import dataclasses
from dataclasses import dataclass
from functools import partial
import json
import sys
from typing import Optional as TOptional # Optional conflicts with pyparsing
from typing import Dict
from typing import TextIO

import pyparsing
from pyparsing import alphas, alphanums, \
    delimitedList, nums, restOfLine, \
    Forward, Keyword, Literal, Optional, Regex, Word, ZeroOrMore

from m5pre import M5Pre


def separated_list(base, separator, allow_term_sep = False):
    l = delimitedList(base, separator)
    if allow_term_sep:
        if type(separator) == str:
            separator = Literal(separator)
        l += Optional(separator.suppress())
    return l

def literal_suppress(s: str):
    return Literal(s).suppress()

def dict_field_no_init(repr = True):
    return dataclasses.field(default_factory = dict, init = False, repr = repr)


def to_camelcase(s: str) -> str:
    return ''.join([x.capitalize() for x in s.split('_')])


class M5MetaError(Exception):
    pass

@dataclass
class Field:
    name:     str
    origin:   TOptional[int] = None
    width:    TOptional[int] = None
    enum:     TOptional[Dict[str, int]] = None
    default:  TOptional[int] = None
    stats:    TOptional[Dict[int, int]] = dataclasses.field(default_factory=Counter, init = False, repr = False)
    enum_rev: TOptional[Dict[int, str]] = dict_field_no_init(repr = False)

    def write_fdef(self, f):
        f.write(f'field {self.name} lsb {self.origin} width {self.width}')
        if self.enum is not None:
            f.write(' enum')
        f.write('\n')
        if self.enum is not None:
            for n, v in self.enum.items():
                f.write(f'  {n} = {v}\n')
            f.write('end\n')
    
    def write_vhdl(self, f):
        f.write(f'  subtype {self.name}_t is std_logic_vector({self.width-1} downto 0);\n')
        if self.enum is not None:
            for n, v in self.enum.items():
                f.write(f'  constant {self.name}_{n}: {self.name}_t := "{v:0{self.width}b}";\n')
        f.write('\n')

@dataclass
class AddressSpace:
    name:     str
    size:     TOptional[int] = None
    width:    TOptional[int] = None
    fields:   Dict[str, Field] = dict_field_no_init()
    macros:   Dict[str, Dict] = dict_field_no_init()
    bits:     bytearray = dataclasses.field(default_factory = bytearray, init = False)
    pc:       int = dataclasses.field(default = 0, init = False)
    inst:     Dict[int, dict] = dict_field_no_init()
    data:     Dict[int, int] = dict_field_no_init()

    def assign_bits(self, width, origin = None):
        if origin is None:
            try:
                origin = self.bits.index(bytearray(width))
            except ValueError:
                origin = len(self.bits)
        if any(self.bits[origin:origin+width]):
            raise M5MetaError(f'some of bits {origin}..{origin+width-1} already assigned in word')
        self.bits[origin:origin+width] = [1]*width
        return origin

    def instruction_to_object(self, addr, instruction):
        #debug = self.name.startswith('dispatch')
        inst = 0
        for fn, fd in self.fields.items():
            if fn in instruction:
                fv = instruction[fn]
            elif fd.default is not None:
                fv = fd.default
            else:
                raise M5MetaError('unassigned field {fn} at address {addr:04x}')
            fd.stats[fv] += 1
            inst |= (fv << fd.origin)
        return inst

    def generate_object(self):
        for addr, inst in self.inst.items():
            self.data[addr] = self.instruction_to_object(addr, inst)

    def write_hex_file(self, fn):
        hex_digits = (self.width + 3)//4
        with open(fn, 'w') as f:
            prev_addr = -1
            for addr in sorted(self.data.keys()):
                if prev_addr is None or addr != prev_addr + 1:
                    print(f'@{addr:04x}', file = f)
                data = self.data[addr]
                hex = '.format('
                print(f'{data:0{hex_digits}x}', file = f)
                prev_addr = addr

    def write_fdef(self, f):
        f.write(f'word width {self.width}\n')
        for fd in self.fields.values():
            f.write('\n')
            fd.write_fdef(f)

    def write_vhdl(self, f, name):
        f.write('library ieee;\n')
        f.write('use ieee.std_logic_1164.all;\n')
        f.write('use ieee.numeric_std.all;\n')
        f.write('\n')
        f.write(f'package {name}_package is\n')
        for fd in self.fields.values():
            fd.write_vhdl(f)
        f.write(f'  type {name}_t is\n')
        f.write(f'    record\n')
        for fd in self.fields.values():
            f.write(f'      {fd.name}: {fd.name}_t;\n')
        f.write(f'    end record;\n')
        f.write(f'\n')
        f.write(f'  type {name}_array_t is array (natural range <> of {name}_t;\n')
        f.write(f'\n')

        f.write(f'end package {name}_package;\n')

class M5Meta:
    def __init__(self, src_file, obj_base_fn):

        self.src_file = src_file
        self.obj_base_fn = obj_base_fn

        self.pass_num = 0

        self.symtab = {}
        self.spaces = {}
        self.space = None

        self.grammar = self.define_grammar()

    def print_production(self, name, x):
        if self.pass_num == 2:
            print(f'{name}: {x}')

    def process_enum_value(self, d, value):
        if type(value) is int:
            return value
        if value in d:
            return d[value]
        raise M5MetaError(f'unknown symbol "{value}" in field enum constant definition')

    def action_field_enum_def(self, x):
        prev = -1
        d = {}
        for i in range(1, len(x)):
            name = x[i][0]
            value = x[i][1]
            if name in d:
                raise M5MetaError(f'multiply defined field enum constant "{name}"')
            if value is None:
                value = prev + 1
            prev = value
            d[name] = self.process_enum_value(d, value)
        return {'enum': d}

    def action_field_bool(self, x):
        d = { 'false': 0, 'true': 1 }
        return {'width': 1,
                'enum': d }

    def action_field_def(self, x):
        name = x[0]
        field = Field(name)
        for k, v in x[1].items():
            setattr(field, k, v)
        if field.default is not None and type(field.default) is str:
            if field.enum is None:
                de = field.default
                raise M5MetaError(f'no definition for default "{de}"')
            else:
                field.default = self.process_enum_value(field.enum, field.default)
        max_value = None
        if field.enum is not None:
            max_value = max(field.enum.values())
        if field.default is not None:
            if max_value is None:
                max_value = field.default
            else:
                max_value = max(max_value, field.default)
        if field.width is None:
            if max_value is None:
                raise M5MetaError(f'field width not specified and cannot be inferred')
            field.width = max_value.bit_length()
        elif max_value is not None and field.width < max_value.bit_length():
            raise M5MetaError(f"field {name} width {field.width} isn't wide enough for enum or default values")
        return field

    def action_field_defs(self, x):
        for space_name in x[1]:
            if space_name not in self.spaces:
                raise M5MetaError(f'unknown address space {space_name}')
            space = self.spaces[space_name]
            for field_def in x[2:]:
                field_name = field_def.name
                if self.pass_num == 1:
                    if field_name in space.fields:
                        raise M5MetaError(f'multiply defined field {field_name} in address space {space_name}')
                    field_def.origin = space.assign_bits(field_def.width, field_def.origin)
                    space.fields[field_name] = field_def
                elif self.pass_num == 2:
                    field = space.fields[field_name]
                    field.default = field_def.default

    def action_macro_def(self, x):
        name = x[1]
        inst = x[2]
        if self.pass_num == 1 and name in self.space.macros:
            raise M5MetaError(f'multiply defined macro {name} in address space {space_name}')
        self.space.macros[name] = inst

    def eval_symbol(self, s, field_name):
        field = self.space.fields[field_name]
        if field.enum is not None and s in field.enum:
            return field.enum[s]
        if s in self.symtab:
            return self.symtab[s]
        if self.pass_num == 1:
            return 0  # may be a forward reference
        raise M5MetaError(f'cannot evaluate "{s}" as a value for field {field_name}')
        return 0

    def add_field(self, fd, fname, fvalue):
        if fname in fd:
            raise M5MetaError(f'field {fname} multiply used in instruction')
        if fname not in self.space.fields:
            raise M5MetaError(f'field {fname} not defined')
        field = self.space.fields[fname]
        if type(fvalue) is str:
            fvalue = self.eval_symbol(fvalue, fname)
        fd[fname] = fvalue

    def action_instruction(self, x):
        fd = { }
        for fa in x:
            if type(fa) is str:
                # macro invocation
                if fa not in self.space.macros:
                    raise M5MetaError(f'undefined macro {fa}')
                for mfn, mfv in self.space.macros[fa].items():
                    self.add_field(fd, mfn, mfv)
            else:
                # field assignment
                self.add_field(fd, fa[0], fa[1])
        return [fd]

    def action_l_instruction(self, x):
        while type(x[0]) is str:
            label = x[0]
            x = x[1:]
            if label in self.symtab:
                if self.space.pc != self.symtab[label]:
                    raise M5MetaError(f'multiply defined symbol "{label}", original value {self.symtab[label]:04x}, new value {self.pc:04x}')
            else:
                self.symtab[label] = self.space.pc
        if self.pass_num == 2:
            fields = x[0]
            if self.space.pc in self.space.inst:
                raise M5MetaError(f'multiple instructions at address space {self.space.name} address {self.space.pc:04x}')
            self.space.inst[self.space.pc] = fields
        self.space.pc += 1

    def action_origin(self, x):
        self.space.pc = x[1]

    def action_space_def(self, x):
        name = x[1]
        attrs = x[2]
        try:
            size = attrs['size']
        except:
            raise M5MetaError(f'no size specified for address space "{name}"')
        try:
            width = attrs['width']
        except:
            raise M5MetaError(f'no width specified for address space "{name}"')
        if self.pass_num == 1:
            if name in self.spaces:
                raise M5MetaError(f'multiply defined address space "{name}"')
            self.space = AddressSpace(name, size, width)
            self.spaces[name] = self.space
        else:
            if name not in self.spaces:
                raise M5MetaError(f'undefined address space "{name}"')
            self.space = self.spaces[name]

    def action_space_select(self, x):
        name = x[1]
        if name not in self.spaces:
            raise M5MetaError(f'undefined address space {name}')
        self.space = self.spaces[name]

    def action_merge_dicts(self, x):
        r = { }
        for d in x:
            r.update(d)  # XXX doesn't complain about duplicates
        return r

    def define_grammar(self):
        dec_int = Word(nums).setParseAction(lambda toks: int(toks[0]))
        hex_int = Regex('0[xX][0-9a-fA-F]*').setParseAction(lambda toks: int(toks[0][2:],16))

        # hexadecimal must precede decimal, or decimal will grab the leading
        # '0' out of the '0x' prefix of a hexadecimal constant
        integer = hex_int | dec_int

        ident = Word(alphas, alphanums + '_')

        ARROW  = literal_suppress('=>')
        COLON  = literal_suppress(':')
        COMMA  = literal_suppress(',')
        EQUALS = literal_suppress('=')
        LBRACE = literal_suppress('{')
        RBRACE = literal_suppress('}')
        SEMI   = literal_suppress(';')

        BOOL     = Keyword('bool')
        DEFAULT  = Keyword('default')
        ENUM     = Keyword('enum')
        EQUATE   = Keyword('equate')
        FIELDS   = Keyword('fields')
        IF       = Keyword('if')
        MACRO    = Keyword('macro')
        ORIGIN   = Keyword('origin')
        SIZE     = Keyword('size')
        SPACE    = Keyword('space')
        WIDTH    = Keyword('width')

        value = ident | integer

        field_bool_attribute = BOOL
        field_bool_attribute.setParseAction(self.action_field_bool)

        field_enum_item = ident + Optional(EQUALS + value)
        field_enum_item.setParseAction(lambda x: [[x[0], x[1] if len(x) > 1 else None]])

        field_assignment = ident + ARROW + value
        field_assignment.setParseAction(lambda x: [[x[0], x[1]]])

        macro_subst = ident
        macro_subst.setParseAction(lambda x: [x[0]])
        
        instruction_part = field_assignment | macro_subst

        instruction = separated_list(instruction_part, COMMA)
        instruction.setParseAction(self.action_instruction)

        label = ident + COLON
        label.setParseAction(lambda x: [x[0]])

        labels = ZeroOrMore(label)

        l_instruction = labels + instruction
        l_instruction.setParseAction(self.action_l_instruction)

        macro_def = MACRO + ident + COLON + LBRACE + instruction + RBRACE
        macro_def.setParseAction(self.action_macro_def)

        field_enum_def = ENUM + LBRACE + separated_list(field_enum_item, SEMI, allow_term_sep = True) + RBRACE
        field_enum_def.setParseAction(self.action_field_enum_def)

        field_origin_attribute = ORIGIN + value
        field_origin_attribute.setParseAction(lambda x: {'origin': x[1]})

        field_width_attribute = WIDTH + value
        field_width_attribute.setParseAction(lambda x: {'width': x[1]})

        field_default = DEFAULT + value
        field_default.setParseAction(lambda x: {'default': x[1]})

        field_attribute = field_origin_attribute | field_width_attribute | field_enum_def | field_bool_attribute | field_default

        field_attributes = separated_list(field_attribute, COMMA)
        field_attributes.setParseAction(self.action_merge_dicts)

        field_def = ident + COLON + field_attributes
        field_def.setParseAction(self.action_field_def)

        address_spaces = separated_list(ident, ',')
        address_spaces.setParseAction(lambda x: [x])

        field_defs = FIELDS + address_spaces + COLON + LBRACE + separated_list(field_def, SEMI, allow_term_sep = True) + RBRACE
        field_defs.setParseAction(self.action_field_defs)

        space_size_attribute = SIZE + value
        space_size_attribute.setParseAction(lambda x: { 'size': x[1] })

        space_width_attribute = WIDTH + value
        space_width_attribute.setParseAction(lambda x: { 'width': x[1] })

        space_attribute = space_size_attribute | space_width_attribute

        space_attributes = separated_list(space_attribute, COMMA)
        space_attributes.setParseAction(self.action_merge_dicts)

        space_def = SPACE + ident + COLON + space_attributes
        space_def.setParseAction(self.action_space_def)

        space_select = SPACE + ident
        space_select.setParseAction(self.action_space_select)

        equate = ident + EQUATE + value

        origin = ORIGIN + value
        origin.setParseAction(self.action_origin)

        statement_list = Forward()

        if_statement = IF + value + LBRACE + statement_list + RBRACE

        statement = space_def | field_defs | space_select | macro_def | equate | origin | if_statement | l_instruction

        statement_list <<= separated_list(statement, ';', allow_term_sep = True)

        comment = Literal('//') + Optional(restOfLine)

        compilation_unit = statement_list
        compilation_unit.ignore(comment)

        #field_def.setParseAction(partial(self.print_production, 'field_def'))

        return compilation_unit

    def write_listing_file(self):
        print(f'word width {self.word_width}, bits unused {self.word_bits_unused}', file = self.listing_file)
        print(f'{self.word_count} words of microcode', file = self.listing_file)
        print(file = self.listing_file)

        for fn, fd in self.space.fields.items():
            msb = fd.origin + fd.width - 1
            lsb = fd.width
            print(f'field {fn} {lsb} {lsb}:', file = self.listing_file)
            for value, count in sorted(fd.stats.items()):
                vstr = ''
                if fd.enum is not None:
                    if fd.enum_rev is None:
                        fd.enum_rev = { v: k for k, v in fd.enum.items() }
                    vstr = fd.enum_rev.get(value, '')
                    if vstr != '':
                        vstr = ' (' + vstr + ')'
                print(f'  value {value}{vstr}: {count}', file = self.listing_file)


    def pass12(self):
        result = self.grammar.parseString(self.src, parseAll = True)

    def pass3(self):
        for name, space in self.spaces.items():
            space.generate_object()
            space.write_hex_file(self.obj_base_fn + '_' + space.name + '.hex')
            with open(self.obj_base_fn + '_' + space.name + '.fdef', 'w') as f:
                space.write_fdef(f)
            if False:
                with open(self.obj_base_fn + '_' + space.name + '.vhdl', 'w') as f:
                    space.write_vhdl(f, name)


    passes = [None,
              pass12,
              pass12,
              pass3]

    def assemble(self,
                 listf: TOptional[TextIO] = None):
        listing_file = listf
        self.src = M5Pre(self.src_file).read()
        for p in range(1, len(self.passes)):
            print(f'pass {p}')
            self.pass_num = p
            self.passes[p](self)
        if listf is not None:
            self.write_listing_file()


def main():
    parser = argparse.ArgumentParser(description = 'Microcode assembler')

    parser.add_argument('asmfile',
                        type = argparse.FileType('r'),
                        help = 'microcode assembler source file')

    parser.add_argument('-l', '--list',
                        type = argparse.FileType('w'),
                        help = 'listing file')

    args = parser.parse_args()

    if args.asmfile.name.endswith('.m5'):
        obj_base_fn = args.asmfile.name[:-3]
    else:
        obj_base_fn = args.asmfile.name
    m5meta = M5Meta(args.asmfile, obj_base_fn = obj_base_fn)
    m5meta.assemble(args.list)
        


if __name__ == '__main__':
    main()
