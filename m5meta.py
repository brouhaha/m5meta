#!/usr/bin/python3

# m5meta: Meta assembler
__version__ = '2.0.0'
__copyright__ = 'Copyright 2019-2023 Eric Smith'
# SPDX-License-Identifier: GPL-3.0-only
__license__ = 'GPL 3.0 only'

# This program is free software: you can redistribute it and/or modify
# it under the terms of version 3 of the GNU General Public License
# as published by the Free Software Foundation.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__author__ = 'Eric Smith'
__email__ = 'spacewar@gmail.com'
__description__ = 'M5 Meta Assembler'
__url__ = 'https://github.com/brouhaha/m5meta',


import sys
__min_python_version__ = '3.11'
if sys.version_info < tuple([int(n) for n in __min_python_version__.split('.')]):
    sys.exit("Python version %s or later is required.\n" % __min_python_version__)


__all__ = ['__version__', '__copyright__', '__author__', '__email__',
           '__license__', '__description__', '__url__',
           'M5Meta', 'M5MetaError']

import argparse
import copy
import dataclasses
from dataclasses import dataclass
from functools import partial
import json
from typing import (Optional as TOptional,       # typing.Optional conflicts with pyparsing.Optional
                    Self,
                    TextIO)

import pyparsing
from pyparsing import (alphas, alphanums,
                       delimitedList, nums, restOfLine,
                       Forward, Keyword, Literal,
                       Optional as POptional,    # pyparsing.Optional conflicts with typing.Optional,
                       Regex, Word, ZeroOrMore)

from m5pre import M5Pre


def is_set_or_dict(obj):
    return type(obj) in frozenset([set, frozenset, dict])


make_set_dict = { set:           (lambda x: x),
                  frozenset:     (lambda x: x),
                  dict:          (lambda x: set(x.keys()))
                 }
def make_set(obj):
    return make_set_dict.get(type(obj), (lambda x: set([x])))(obj)

def merge_dicts(self,
                ld: list[dict]):
    r = { }
    for d in ld:
        r.update(d)  # XXX doesn't complain about duplicates
    return r


def separated_list(base, separator, allow_term_sep = False):
    l = delimitedList(base, separator)
    if allow_term_sep:
        if type(separator) == str:
            separator = Literal(separator)
        l += POptional(separator.suppress())
    return l

def literal_suppress(s: str):
    return Literal(s).suppress()


class DictDeepCompare:
    def __init__(self, dict = None, /, **kwargs):
        UserDict.__init__(self, dict = dict, **kwargs)

        

def dict_field_no_init(repr = True):
    return dataclasses.field(default_factory = dict, init = False, repr = repr)


def get_attrs(attrs, attr_names):
    if attrs is None:
        attrs = { }
    results = [None] * len(attr_names)
    for i in range(len(attr_names)):
        if attr_names[i] in attrs:
            results[i] = attrs[attr_names[i]]
    for n in attrs.keys():
        if n not in attr_names:
            raise M5MetaError(f'inappropriate attribute {n}')
    return results


class M5MetaError(Exception):
    pass


# Making a dataclass a subclass of dict or collections.UserDict
# seems to cause mysterious problems; the data dictionary ends
# up containing an element { dict: None }, and if deleted, it
# just comes back. So instead, do some of the equivalent of
# UserClass here.
# Only the symbol table is a bare M5Type. All others are
# subclasses.
@dataclass
class M5Type:
    # these are not applicable for symbol table root
    name:     TOptional[str] = None
    width:    TOptional[int] = None
    origin:   TOptional[int] = None  # relative to origin of parent

    def __post_init__(self):
        self.__data = {}

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, key):
        if key in self.__data:
            return self.__data[key]
        raise KeyError(key)

    def __setitem__(self, key, item):
        self.__data[key] = item
        # WARNING: special-case behavior not like a normal dictionary
        # insert into child a reference to the parent
        item.parent = self

    def __delitem__(self, key):
        del self.__data[key]

    def __iter__(self):
        return iter(self.__data)

    def __contains__(self, key):
        return key in self.__data

    def get(self, key, default=None):
        if key in self.__data:
            return self.__data[key]
        return default

    # for deep comparison
    def __eq__(self, other):
        if len(self) != len(other):
            return False
        if sorted(self.keys()) != sorted(other.keys()):
            return False
        for n, v in self.items():
            if v != other[n]:
                return False
        return True

    def print(self,
              indent: int = 0,
              recurse: bool = True,
              file: TextIO = sys.stdout,
              root: Self = None,
              name_in_parent: str = None):
        if (root is None or
            root is self):
            root = self
            print('symbol table:', file = file)
        else:
            if name_in_parent is None:
                name_in_parent = '<anon>'
            print(f'{" "*indent}{name_in_parent}: {type(self).__name__} (', file = file, end = '')
            comma = ''
            for f in dataclasses.fields(self):
                fn = f.name
                fv = getattr(self, fn)
                print(f'{comma}{fn}={fv}', file = file, end = '')
                comma = ', '
            print(')', file = file)
        if recurse:
            for k, v in self.__data.items():
                do_recurse = (recurse and
                              ((self is root) or
                               (v.name not in root) or
                               (v != root[v.name])))
                #print(f'{" "*(indent+2)} {k}', file = file, end = '')
                if do_recurse:
                    v.print(indent = indent+2,
                            recurse = do_recurse,
                            file = file,
                            root = root,
                            name_in_parent = k)
                else:
                    print(file = file)

    def get_typed_entry(self,
                        requested_classes: type(Self) | set[type(Self)],
                        name: str,
                        exist: bool = True):
        requested_classes = make_set(requested_classes)
        try:
            v = self[name]
        except:
            if exist:
                raise M5MetaError(f'"{name}" is not defined')
            return None
        if not exist:
            if type(v) in requested_classes:
                raise M5MetaError(f'"{name}" is already defined as a {type(v)}')
            raise M5MetaError(f'"{name}" is already defined, but not as any of {requested_classes}')
        if type(v) not in requested_classes:
            raise M5MetaError(f'"{name}" is not any of {requested_classes}')
        return v

    def get_all_typed_entries(self,
                              requested_class: type(Self)):
        for v in self.__data:
            if isinstance(v, requested_class):
                yield v

    def set_attrs(self, attrs):
        get_attrs(attrs, [])    # no defined attributes

    def deep_equal(self, other):
        if not self.__eq__(other):
            return False
        for n, v in self.items():
            if (n not in other or
                self[n] != other[n]):
                return False
        for n, v in other.items():
            if n not in self:
                return False
        return True

    def compute_width(self):
        width = 0
        for n, c in self.__data.items():
            if c.origin is None:
                c_origin = 0
            else:
                c_origin = c.origin
            if c.width is None:
                c.compute_width()
                if c.width is None:
                    raise M5MetaError(f'cannot determine width of "{c.name}"')
            width = max(width, c_origin + c.width)
        if self.width is None:
            self.width = width
        else:
            if self.width < width:
                raise M5MetaError(f'width attribute {self.width} less than computed minimum {width}')


@dataclass
class M5Integer(M5Type):
    value:  TOptional[int] = None
    signed: bool = False

    def set_attrs(self, attrs):
        print(f'set_attrs {attrs=}')
        origin, width, default = get_attrs(attrs, ['origin', 'width', 'default'])  # XXX should only allow origin and default if this is a field in a struct

    def compute_width(self):
        print(f'M5Integer.compute_width(), {self=}')
        if self.value is None:
            return
        if self.value < 0 and not self.signed:
            raise M5MetaError(f'unsigned has negative value')
        width = self.value.bit_length()
        if self.signed:
            self.width += 1
        if self.width is None:
            self.width = width
        elif self.width < width:
            raise M5MetaError(f'width attribute {self.width} less than computed minimum {width}')

    def __post_init__(self):
        super().__post_init__()
        self.compute_width()


@dataclass
class M5String(M5Type):
    value:  TOptional[str] = None


@dataclass
class M5Enum(M5Type):
    max_value: int = 0

    def define_value(self,
                     name: str,
                     value: TOptional[int] = None):
        if name in self and self[name] is not None:
            raise M5MetaError(f'duplicate enum name {name}')
        if value is None:
            value = self.max_value
        if self.width is not None and value.bit_length() > self.width:
            raise M5MetaError(f'enum value {name}={value} wider than enum declared width {self.width}')
        self.max_value = max(value+1, self.max_value)
        print(f'define_value: {name=} {value=}')
        self[name] = M5Integer(name = name, value = value, signed = value < 0)

    def set_attrs(self, attrs):
        width = get_attrs(attrs, ['width'])[0]
        if width is not None:
            self.width = width


@dataclass
class M5Struct(M5Type):
    union:    bool = False


# used for individual instructions as well as macros
@dataclass
class M5Instruction(M5Type):
    pass


# address space dict is empty
@dataclass
class M5AddressSpace(M5Type):
    depth:    TOptional[int] = None
    struct:   TOptional[M5Struct] = None
    bits:     bytearray = dataclasses.field(default_factory = bytearray, init = False)
    pc:       int = dataclasses.field(default = 0, init = False)
    inst:     dict[int, dict] = dict_field_no_init()
    obj:      dict[int, int] = dict_field_no_init()

    def print(self,
              indent: int = 0,
              recurse: bool = True,
              file: TextIO = sys.stdout,
              root: Self = None,
              name_in_parent: str = None):
        super().print(indent = indent,
                      recurse = recurse,
                      file = file,
                      root = root,
                      name_in_parent = name_in_parent)
        if self.struct is None or isinstance(self.struct, str):
            return
        self.struct.print(indent = indent+2,
                          recurse = recurse,
                          file = file,
                          root = root,
                          name_in_parent = None)

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
        for fn, ft in self.struct:
            print(f'instruction_to_object {fn=}, {ft=}')
            if fn in instruction:
                fv = instruction[fn]
            elif ft.default is not None:
                fv = ft.default
            else:
                raise M5MetaError('unassigned field {fn} at address {addr:04x}')
            #fd.stats[fv] += 1
            inst |= (fv << ft.origin)
        return inst

    def generate_object(self):
        for addr, inst in self.inst.items():
            print(f'generate_object {addr=} {inst=}')
            self.obj[addr] = self.instruction_to_object(addr, inst)

    def write_hex_file(self, fn):
        hex_digits = (self.width + 3)//4
        with open(fn, 'w') as f:
            prev_addr = -1
            for addr in sorted(self.obj.keys()):
                if prev_addr is None or addr != prev_addr + 1:
                    print(f'@{addr:04x}', file = f)
                d = self.obj[addr]
                hex = '.format('
                print(f'{d:0{hex_digits}x}', file = f)
                prev_addr = addr


class M5Meta:
    def __init__(self, src_file, obj_base_fn):

        self.src_file = src_file
        self.obj_base_fn = obj_base_fn

        self.pass_num = 0

        self.symtab = M5Type()

        self.type_being_defined = None
        self.type_being_defined_stack = []

        self.space = None

        self.anon_number = 0

        self.grammar = self.define_grammar()

    def get_symbol(self, name):
        if name in self.symtab:
            return self.symtab[name]
        if (self.space is not None and
            isinstance(self.space.struct, M5Struct) and
            name in self.space.struct):
            return self.space.struct[name]
        raise M5MetaError(f'undefined symbol "{name}"')

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
        print(f'action_enum_item {x=}')
        if not hasattr(self, 'enum_being_defined'):
            self.enum_being_defined = M5Enum()
        name = x[0]
        value = x[1] if len(x) > 1 else None
        self.enum_being_defined.define_value(name, value)

    def action_enum_def(self, x):
        print(f'action_enum_def {x=}')
        name = x[1]
        attrs = x[2]
        if name is None:
            raise M5MetaError('anonymous enum definition not embedded')
            #name = '___enum_' + str(self.anon_number)
            #self.anon_number += 1
        self.enum_being_defined.name = name
        self.enum_being_defined.set_attrs(attrs)
        print(f'action_enum_def {self.enum_being_defined}')
        self.enum_being_defined.compute_width()
        print(f'action_enum_def {self.enum_being_defined}')
        if name in self.symtab and self.enum_being_defined != self.symtab[name]:
            raise M5MetaError(f'type name "{name}" redefined')
        self.symtab[name] = self.enum_being_defined
        delattr(self, 'enum_being_defined')

    def action_type_integer(self, x):
        print(f'action_type_integer {x=}')
        t = M5Integer(signed = (x[0] == 'integer'))
        return [t]

    def action_item_type(self, x):
        print(f'action_item_type {x=}')

    def action_item_named(self, x):
        print(f'action_item_named {x=}')
        t, attrs, name = x
        if isinstance(t, str):
            t = copy.copy(self.symtab[t])  # shallow copy because we're likely to change some attributes
            # XXX check for enum, struct
        print(f'action_item_named {name=} {t=} {attrs=}')
        t.set_attrs(attrs)
        print(f'action_item_named: setting child "{name}" of {id(self.type_being_defined)}')
        self.type_being_defined[name] = t
        print(f'{self.type_being_defined=}')
        return[t]

    def action_struct_or_union(self, x):
        print(f'action_struct_or_union {x=}')
        if self.type_being_defined:
            self.type_being_defined_stack.append(self.type_being_defined)
        self.type_being_defined = M5Struct(name = None,
                                           union = x[0] == 'union')
        print(f'action_struct_or_union: created {id(self.type_being_defined)}')

    def action_struct_def(self, x):
        print(f'action_struct_def {x=}')
        name = x[1]

        if name is None:
            print(f'action_struct_def: no name')
            result = self.type_being_defined
        else:
            # XXX should check for duplicate definition
            self.type_being_defined.name = name
            self.symtab[name] = self.type_being_defined
            result = name

        print(f'action_struct_def done {id(self.type_being_defined)} {self.type_being_defined}')

        self.type_being_defined = None
        if len(self.type_being_defined_stack):
            self.type_being_defined = self.type_being_defined_stack.pop()

        return result
        

    def action_macro_def(self, x):
        name = x[1]
        inst = x[2]
        if self.pass_num == 1:
            self.symtab.get_typed_entry(M5Instruction, name, False)  # make sure no macro of this name already exists
        else:
            self.symtab.get_typed_entry(M5Instruction, name)  # make sure it exists
            self.symtab[name] = inst              # replace, because it may have contained forward reference

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
        print(f'action_instruction {x=}')
        inst = M5Instruction()
        for part in x:
            name = part[0]
            if len(part) == 1:
                # macro invocation
                macro = self.symtab.get_typed_entry(M5Instruction, name)
                for fn, fv in t:
                    inst[fn] = fv
            elif len(part) == 2:
                # field assignment
                value = part[1]
                print(f'field assignment {name=} {value=}')
                struct_elem = self.space.struct.get_typed_entry(M5Integer, name)  # XXX need to handle other types
                if isinstance(value, int):
                    value = M5Integer(value = part[1])
                elif isinstance(part[1], str):
                    value = self.symtab.get_typed_entry(M5Integer, name)
                else:
                    M5MetaError(f'cannot resolve field "{value}"')
                inst[name] = value
        return [inst]

    def action_l_instruction(self, x):
        while type(x[0]) is str:
            label = x[0]
            x = x[1:]
            if label in self.symtab:
                ov = self.symtab.get_typed_entry(M5Integer, label).value
                if self.space.pc != ov:
                    raise M5MetaError(f'multiply defined symbol "{label}", original value {ov:04x}, new value {self.space.pc:04x}')
            else:
                self.symtab[label] = M5Integer(value = self.space.pc)
        if self.pass_num == 2:
            fields = x[0]
            if self.space.pc in self.space.inst:
                raise M5MetaError(f'multiple instructions at address space {self.space.name} address {self.space.pc:04x}')
            self.space.inst[self.space.pc] = fields
        self.space.pc += 1

    def action_origin_statement(self, x):
        v = x[1]
        self.space.pc = v

    def action_equate_statement(self, x):
        print(f'action_equate_statement {x=}')
        name, _, value = x
        if self.pass_num == 1:
            self.symtab.get_typed_entry(M5Integer, name, False)  # make sure it doesn't already exist
            self.symtab[name] = M5Integer(value = value)
        

    def action_space_def(self, x):
        attrs = x[1]
        struct = x[2]
        name = x[3]
        print(f'action_space_def {attrs=}')
        print(f'action_space_def {struct=}')
        print(f'action_space_def {name=}')
        width, depth = get_attrs(attrs, ['width', 'depth'])
        if isinstance(struct, str):
            a = self.symtab.get_typed_entry(M5Struct, struct)  # make sure struct exists
        if self.pass_num == 1:
            self.symtab.get_typed_entry(M5AddressSpace, name, exist = False)  # address space must not have been previously defined
            self.symtab[name] = M5AddressSpace(name = name,
                                               width = width,
                                               depth = depth,
                                               struct = struct)
        else:
            a = self.symtab.get_typed_entry(M5AddressSpace, name)
            # XXX compare address space

    def action_space_ident(self, x):
        print(f'action_space_content {x=}')
        name = x[1]
        self.space = self.symtab.get_typed_entry(M5AddressSpace, name)

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
#        width_attribute = POptional(WIDTH, None) + value
        width_attribute.set_parse_action(self.action_width_attribute)

        origin_attribute = ORIGIN + value
        origin_attribute.set_parse_action(self.action_origin_attribute)

        attribute = depth_attribute | default_attribute | width_attribute | origin_attribute

        attributes = LPAREN + separated_list(attribute,
                                             COMMA,
                                             allow_term_sep = False) + RPAREN
        attributes.set_parse_action(merge_dicts)

        enum_item = ident + POptional(EQUALS + value)
        enum_item.set_parse_action(self.action_enum_item)

        enum_item_list = LBRACE + separated_list(enum_item,
                                                 SEMI,
                                                 allow_term_sep = True) + RBRACE

        enum_def = ENUM + POptional(ident, None) + POptional(attributes, None) + enum_item_list
        enum_def.set_parse_action(self.action_enum_def)


        signed_integer_type = INTEGER

        unsigned_integer_type = UNSIGNED + POptional(INTEGER)

        integer_type = signed_integer_type | unsigned_integer_type

        type_integer = integer_type
        type_integer.set_parse_action(self.action_type_integer)


        type_enum = enum_def

        struct_def = Forward()
        type_struct = struct_def

        type_named = ident

        item_type = type_integer | type_enum | type_struct | type_named
        item_type.set_parse_action(self.action_item_type)

        item_named = item_type + POptional(attributes, None) + ident
        item_named.set_parse_action(self.action_item_named)

        item_list = LBRACE + separated_list(item_named,
                                            SEMI,
                                            allow_term_sep = True) + RBRACE

        struct_or_union = ( STRUCT | UNION )
        struct_or_union.set_parse_action(self.action_struct_or_union)

        struct_def = struct_or_union + POptional(ident, None) + POptional(attributes, None) + item_list
        struct_def.set_parse_action(self.action_struct_def)

        space_def = SPACE + POptional(attributes, None) + (struct_def | ident) + ident
        space_def.set_parse_action(self.action_space_def)

        equate_statement = ident + EQUATE + value
        equate_statement.set_parse_action(self.action_equate_statement)

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

        def_statement = equate_statement | space_def | enum_def | struct_def | macro_def | space_content

        def_statement_list = separated_list(def_statement, ';', allow_term_sep = True)

        comment = Literal('//') + POptional(restOfLine)

        compilation_unit = def_statement_list
        compilation_unit.ignore(comment)

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

    def write_symtab(self):
        self.symtab.print(root = self.symtab)
        #for k, v in self.symtab:
        #    print(f'{k:12s} {v}')


    def pass12(self):
        result = self.grammar.parseString(self.src, parseAll = True)

    def pass3(self):
        for space in self.symtab.get_all_typed_entries(M5AddressSpace):
            print(f'outputting object for address space {space.name}')
            space.generate_object()
            space.write_hex_file(self.obj_base_fn + '_' + space.name + '.hex')

            #self.write_symtab()

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
            self.write_symtab()
            return
        if listf is not None:
            self.write_listing_file()
        self.write_symtab()


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
