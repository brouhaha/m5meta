#!/usr/bin/python3

# m5meta: Meta assembler
# Copyright 2023 Eric Smith <spacewar@gmail.com>
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
class M5Type:
    name:    TOptional[str] = None
    width:   TOptional[int] = None
    origin:  TOptional[int] = None


@dataclass
class M5Integer(M5Type):
    pass


@dataclass
class M5EnumValue:
    name:    str
    value:   TOptional[int]


@dataclass
class M5Enum(M5Type):
    values:    Dict[str, M5EnumValue] = dict_field_no_init()
    max_value: int = 0
    width:     TOptional[int] = None

    def define_value(self,
                     name: str,
                     value: TOptional[int] = None):
        if name in self.values and self.values[name] is not None:
            raise M5MetaError(f'duplicate enum name {name}')
        if value is None:
            value = self.max_value + 1
        if self.width is not None and value.bit_length() > self.width:
            raise M5MetaError(f'enum value {name}={value} wider than enum declared width {self.width}')
        self.max_value = max(value+1, self.max_value)
        self.values[name] = M5EnumValue(name, value)

    def assign_width(self):
        if self.width is not None:
            return
        self.width = 0
        for n, v in self.values.items():
            self.width = max(self.width, v.value.bit_length())

    def __eq__(self, other):
        if (self.name != other.name or
            self.max_value != other.max_value or
            self.width != other.width or
            len(self.values) != len(other.values)):
            print(f'{self=}')
            print(f'{other=}')
            return False
        for n, v in self.values.items():
            if (n not in other.values or
                self.values[n] != other.values[n]):
                print(f'{self=}')
                print(f'{other=}')
                return False
        return True


@dataclass
class M5Struct(M5Type):
    union:    bool = False
    fields:   Dict[str, M5Type] = dict_field_no_init()


@dataclass
class AddressSpace:
    name:     str
    width:    TOptional[int] = None
    depth:    TOptional[int] = None
    struct:   TOptional[M5Struct] = None
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
        inst = 0
        for fn, ft in self.struct.fields:
            if fn in instruction:
                fv = instruction[fn]
            elif self.struct.fields[fn].default is not None:
                fv = self.struct.fields[fn].default
            else:
                raise M5MetaError('unassigned field {fn} at address {addr:04x}')
            #fd.stats[fv] += 1
            inst |= (fv << ft.origin)
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

        self.types = {}
        self.type_being_defined = None
        self.type_being_defined_stack = []

        self.space = None

        self.anon_number = 0

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

    def action_width_attribute(self, x):
        print(f'action_width_attribute {x=}')
        return { 'width': x[1] }

    def action_default_attribute(self, x):
        print(f'action_default_attribute {x=}')
        return { 'default': x[1] }

    def action_depth_attribute(self, x):
        print(f'action_depth_attribute {x=}')
        return { 'depth': x[1] }

    def action_origin_attribute(self, x):
        print(f'action_origin_attribute {x=}')
        return { 'origin': x[1] }

    def action_enum_item(self, x):
        if not hasattr(self, 'enum_being_defined'):
            self.enum_being_defined = M5Enum()
        name = x[0]
        value = x[1] if len(x) > 1 else None
        self.enum_being_defined.define_value(name, value)

    def action_enum_width_attribute(self, x):
        # XXX this now has to be done differently!
        width = x[1]
        if not hasattr(self, 'enum_being_defined'):
            self.enum_being_defined = M5Enum()
        if (self.enum_being_defined.width is not None and
            width != self.enum_being_defined.width):
            raise M5MetaError('multiple width attributes on enum')
        self.enum_being_defined.width = width
        return width

    def action_enum_def(self, x):
        name = x[1]
        if name is None:
            raise M5MetaError('anonymous enum definition not embedded')
            #name = '___enum_' + str(self.anon_number)
            #self.anon_number += 1
        self.enum_being_defined.name = name
        self.enum_being_defined.assign_width()
        if name in self.types and self.enum_being_defined != self.types[name]:
            raise M5MetaError(f'type name "{name}" redefined')
        self.types[name] = self.enum_being_defined
        delattr(self, 'enum_being_defined')

    def action_type_integer(self, x):
        print(f'action_type_integer {x=}')

    def action_type_enum(self, x):
        print(f'action_type_enum {x=}')

    def action_type_struct(self, x):
        print(f'action_type_struct {x=}')

    def action_type_named(self, x):
        print(f'action_type_named {x=}')

    def action_item_type(self, x):
        print(f'action_item_type {x=}')

    def action_item_named(self, x):
        print(f'action_item_named {x=}')

    def action_struct_or_union(self, x):
        print(f'action_struct_or_union {x=}')
        if self.type_being_defined:
            self.type_being_defined_stack.append(self.type_being_defined)
        self.type_being_defined = M5Struct(name = None,
                                           union = x[0] == 'union')

    def action_struct_def(self, x):
        print(f'action_struct_def {x=}')
        name = x[1]

        # XXX should check for duplicate definition
        if name:
            self.types[name] = self.type_being_defined
            result = name
        else:
            result = self.type_being_defined

        self.type_being_defined = None
        if len(self.type_being_defined_stack):
            self.type_being_defined = self.type_being_defined_stack.pop()

        return result
        

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

    def action_instruction(self, x):
        fd = { }
        for fa in x:
            if type(fa) is str:
                # macro invocation
                if self.space is None:
                    raise M5MetaError(f'undefined space, so no macro {fa}')
                if fa not in self.space.macros:
                    raise M5MetaError(f'undefined macro {fa}')
                #for mfn, mfv in self.space.macros[fa].items():
                #    self.add_field(fd, mfn, mfv)
            else:
                # field assignment
                #self.add_field(fd, fa[0], fa[1])
                pass
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

    def action_origin_statement(self, x):
        self.space.pc = x[1]

    @staticmethod
    def get_attrs(attrs, attr_names):
        results = []
        for n in attr_names:
            try:
                results.append(attrs[n])
            except:
                raise M5MetaError(f'missing required attribute {n}')
        for n in attrs.keys():
            if n not in attr_names:
                raise M5MetaError(f'inappropriate attribute {n}')
        return results
        
    def action_space_def(self, x):
        attrs = x[1]
        struct = x[2]
        name = x[3]
        print(f'action_space_def {attrs=}')
        print(f'action_space_def {struct=}')
        print(f'action_space_def {name=}')
        width, depth = self.get_attrs(attrs, ['width', 'depth'])
        if isinstance(struct, str):
            try:
                struct = self.types[struct]
            except:
                raise M5MetaError(f'type "{struct}" is undefined')
        if self.pass_num == 1:
            if name in self.spaces:
                raise M5MetaError(f'multiply defined address space "{name}"')
            self.spaces[name] = AddressSpace(name, width, depth, struct)
        else:
            if name not in self.spaces:
                raise M5MetaError(f'undefined address space "{name}"')
            # XXX compare address space

    def action_space_ident(self, x):
        print(f'action_space_content {x=}')
        name = x[1]
        try:
            self.space = self.spaces[name]
        except:
            raise M5MetaError(f'undefined address space "{name}"')

    def action_merge_dicts(self, x):
        r = { }
        for d in x:
            r.update(d)  # XXX doesn't complain about duplicates
        return r

    def define_grammar(self):
        dec_int = Word(nums).set_parse_action(lambda toks: int(toks[0]))
        hex_int = Regex('0[xX][0-9a-fA-F]*').set_parse_action(lambda toks: int(toks[0][2:],16))

        # hexadecimal must precede decimal, or decimal will grab the leading
        # '0' out of the '0x' prefix of a hexadecimal constant
        integer = hex_int | dec_int

        ident = Word(alphas, alphanums + '_')

        ARROW  = literal_suppress('=>')
        COLON  = literal_suppress(':')
        COMMA  = literal_suppress(',')
        EQUALS = literal_suppress('=')
        LPAREN = literal_suppress('(')
        RPAREN = literal_suppress(')')
        LBRACE = literal_suppress('{')
        RBRACE = literal_suppress('}')
        SEMI   = literal_suppress(';')

        DEFAULT  = Keyword('default')
        DEPTH    = Keyword('depth')
        ENUM     = Keyword('enum')
        EQUATE   = Keyword('equate')
        FIELDS   = Keyword('fields')
        IF       = Keyword('if')
        INTEGER  = Keyword('integer')
        MACRO    = Keyword('macro')
        ORIGIN   = Keyword('origin')
        SPACE    = Keyword('space')
        STRUCT   = Keyword('struct')
        UNION    = Keyword('union')
        UNSIGNED = Keyword('unsigned')
        WIDTH    = Keyword('width')

        value = ident | integer

        field_assignment = ident + ARROW + value
        field_assignment.set_parse_action(lambda x: [[x[0], x[1]]])

        macro_subst = ident
        macro_subst.set_parse_action(lambda x: [x[0]])
        
        instruction_part = field_assignment | macro_subst

        instruction = separated_list(instruction_part, COMMA)
        instruction.set_parse_action(self.action_instruction)

        label = ident + COLON
        label.set_parse_action(lambda x: [x[0]])

        labels = ZeroOrMore(label)

        l_instruction = labels + instruction
        l_instruction.set_parse_action(self.action_l_instruction)

        macro_def = MACRO + ident + COLON + LBRACE + instruction + RBRACE
        macro_def.set_parse_action(self.action_macro_def)


        depth_attribute = DEPTH + value
        depth_attribute.set_parse_action(self.action_depth_attribute)
        
        default_attribute = DEFAULT + value
        default_attribute.set_parse_action(self.action_default_attribute)
        
        width_attribute = WIDTH + value
#        width_attribute = Optional(WIDTH, None) + value
        width_attribute.set_parse_action(self.action_width_attribute)

        origin_attribute = ORIGIN + value
        origin_attribute.set_parse_action(self.action_origin_attribute)

        attribute = depth_attribute | default_attribute | width_attribute | origin_attribute

        attributes = LPAREN + separated_list(attribute,
                                             COMMA,
                                             allow_term_sep = False) + RPAREN
        attributes.set_parse_action(self.action_merge_dicts)

        enum_item = ident + Optional(EQUALS + value)
        enum_item.set_parse_action(self.action_enum_item)

        enum_item_list = LBRACE + separated_list(enum_item,
                                                 SEMI,
                                                 allow_term_sep = True) + RBRACE

        enum_def = ENUM + Optional(ident, None) + Optional(attributes, None) + enum_item_list
        enum_def.set_parse_action(self.action_enum_def)


        signed_integer_type = INTEGER

        unsigned_integer_type = UNSIGNED + Optional(INTEGER)

        integer_type = signed_integer_type | unsigned_integer_type

        type_integer = integer_type
        type_integer.set_parse_action(self.action_type_integer)


        type_enum = enum_def

        struct_def = Forward()
        type_struct = struct_def

        type_named = ident

        item_type = type_integer | type_enum | type_struct | type_named
        item_type.set_parse_action(self.action_item_type)

        item_named = item_type + Optional(attributes, None) + ident
        item_named.set_parse_action(self.action_item_named)

        item_list = LBRACE + separated_list(item_named,
                                            SEMI,
                                            allow_term_sep = True) + RBRACE

        struct_or_union = ( STRUCT | UNION )
        struct_or_union.set_parse_action(self.action_struct_or_union)

        struct_def = struct_or_union + Optional(ident, None) + Optional(attributes, None) + item_list
        struct_def.set_parse_action(self.action_struct_def)

        space_def = SPACE + Optional(attributes, None) + (struct_def | ident) + ident
        space_def.set_parse_action(self.action_space_def)

        equate_statement = ident + EQUATE + value

        origin_statement = ORIGIN + value
        origin_statement.set_parse_action(self.action_origin_statement)

        space_statement_list = Forward()

        if_statement = IF + value + LBRACE + space_statement_list + RBRACE

        space_statement = equate_statement | origin_statement | if_statement | l_instruction

        space_statements = LBRACE + separated_list(space_statement,
                                                   SEMI,
                                                   allow_term_sep = True) + RBRACE

        space_ident = SPACE + ident
        space_ident.set_parse_action(self.action_space_ident)

        space_content = space_ident + space_statements

        def_statement = space_def | enum_def | struct_def | macro_def | space_content

        def_statement_list = separated_list(def_statement, ';', allow_term_sep = True)

        comment = Literal('//') + Optional(restOfLine)

        compilation_unit = def_statement_list
        compilation_unit.ignore(comment)

        #field_def.set_parse_action(partial(self.print_production, 'field_def'))

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
        try:
            for p in range(1, len(self.passes)):
                print(f'pass {p}')
                self.pass_num = p
                self.passes[p](self)
        except M5MetaError as e:
            print(f'Error: {e}')
            return
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
