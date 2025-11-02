"""
Compatibility layer providing ``google.protobuf.internal.builder`` for environments
where the upstream package omits the helper (older SentencePiece protobuf stubs expect it).

This module mirrors the implementation shipped with protobuf 3.x so generated *_pb2.py
files can import ``builder`` without adjusting the global installation.
"""

from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import enum_type_wrapper

_sym_db = _symbol_database.Default()


def BuildMessageAndEnumDescriptors(file_des, module):
    """Populate message and enum descriptors on the generated module."""

    def _build_nested_descriptors(msg_des, prefix):
        for name, nested_msg in msg_des.nested_types_by_name.items():
            module_name = prefix + name.upper()
            module[module_name] = nested_msg
            _build_nested_descriptors(nested_msg, module_name + "_")
        for enum_des in msg_des.enum_types:
            module[prefix + enum_des.name.upper()] = enum_des

    for name, msg_des in file_des.message_types_by_name.items():
        module_name = "_" + name.upper()
        module[module_name] = msg_des
        _build_nested_descriptors(msg_des, module_name + "_")


def BuildTopDescriptorsAndMessages(file_des, module_name, module):
    """Create message classes and top-level descriptors on the generated module."""

    def _build_message(msg_des):
        attrs = {}
        for name, nested_msg in msg_des.nested_types_by_name.items():
            attrs[name] = _build_message(nested_msg)
        attrs["DESCRIPTOR"] = msg_des
        attrs["__module__"] = module_name
        message_class = _reflection.GeneratedProtocolMessageType(
            msg_des.name,
            (_message.Message,),
            attrs,
        )
        _sym_db.RegisterMessage(message_class)
        return message_class

    for name, enum_des in file_des.enum_types_by_name.items():
        module["_" + name.upper()] = enum_des
        module[name] = enum_type_wrapper.EnumTypeWrapper(enum_des)
        for enum_value in enum_des.values:
            module[enum_value.name] = enum_value.number

    for name, extension_des in file_des.extensions_by_name.items():
        module[name.upper() + "_FIELD_NUMBER"] = extension_des.number
        module[name] = extension_des

    for name, service in file_des.services_by_name.items():
        module["_" + name.upper()] = service

    for name, msg_des in file_des.message_types_by_name.items():
        module[name] = _build_message(msg_des)


def BuildServices(file_des, module_name, module):
    """Expose service stubs for generated proto files."""
    # Imported lazily to avoid circular dependencies in environments lacking service modules.
    from google.protobuf import service as _service  # pylint: disable=g-import-not-at-top
    from google.protobuf import service_reflection  # pylint: disable=g-import-not-at-top

    for name, service in file_des.services_by_name.items():
        module[name] = service_reflection.GeneratedServiceType(
            name,
            (_service.Service,),
            {"DESCRIPTOR": service, "__module__": module_name},
        )
        stub_name = name + "_Stub"
        module[stub_name] = service_reflection.GeneratedServiceStubType(
            stub_name,
            (module[name],),
            {"DESCRIPTOR": service, "__module__": module_name},
        )


__all__ = [
    "BuildMessageAndEnumDescriptors",
    "BuildTopDescriptorsAndMessages",
    "BuildServices",
]
