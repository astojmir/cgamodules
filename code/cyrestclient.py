# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------
# This is free and unencumbered software released into the public domain.

# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.

# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

# For more information, please refer to <http://unlicense.org/>
# -----------------------------------------------------------------------

#
# Code author:  Aleksandar Stojmirovic
#

import os
import requests
import json
from collections import defaultdict


DEFAULT_HOST = 'http://localhost:'
DEFAULT_PORT = 1234
HEADERS = {'Content-Type': 'application/json',
           'Accept': 'application/json'}

ID = 'id'
NAME = 'name'
DATA = 'data'


def map_property_value_type(value):
    """
    Map value type to a general Java type.

    May perform various conversions or checks
    """
    if isinstance(value, bool):
        # Turn booleans into integers because the conversion in Cytoscape
        # doesn't turn out right
        value = int(value)
        data_type = 'Integer'
        newval = "%d" % value
    elif isinstance(value, int):
        data_type = 'Long'
        newval = "%d" % value
    elif isinstance(value, float):
        data_type = 'Double'
        newval = "%.6g" % value
    elif isinstance(value, str):
        data_type = 'String'
        newval = value
    else:
        # data_type = 'String'
        # newval = str(value)
        raise ValueError("Invalid value type: %s (%s)" %
                         (str(value), str(type(value))))

    return newval, data_type


class CytoNetworkProps:
    """
    Representation of a graph for communicating with CyREST API.

    It is meant to be minimal, sufficient for present needs and extensible.
    """

    node_view_attrs = ['position', 'selected']

    def __init__(self, name='New Network'):

        self.data = {}
        self.elements = {'nodes': [],
                         'edges': []}
        self._nodeid2props = {}
        self.set_net_attr(NAME, name)

    def to_dict(self):
        """
        Dictionary representation suitable for conversion into JSON object
        """
        return dict((k, v) for k, v in self.__dict__.items()
                    if k != '_nodeid2props')

    @classmethod
    def from_dict(cls, d):
        """
        Create new instance from a dictionary representation.
        """
        self = object.__new__(cls)
        self.__dict__.update(d)
        self._nodeid2props = {}
        for obj in self.elements['nodes']:
            _node_id = obj[DATA][ID]
            _node_suid = int(_node_id)
            _name = obj[DATA][NAME]
            self._nodeid2props[_node_suid] = obj
            self._nodeid2props[_node_id] = obj
            self._nodeid2props[_name] = obj
        return self

    @classmethod
    def from_CyRESTNetwork(cls, network_suid, host=DEFAULT_HOST,
                           port=DEFAULT_PORT):
        """
        Query Cytoscape REST API and retrieve a network given by network_suid
        """
        client = CytoscapeClient(host, port)
        d = client.get_network(network_suid)
        return cls.from_dict(d)

    @classmethod
    def from_CyRESTView(cls, network_suid, host=DEFAULT_HOST, port=DEFAULT_PORT):
        """
        Query Cytoscape REST API and retrieve the first network view given by
        network_suid
        """
        client = CytoscapeClient(host, port)
        d = client.get_first_view(network_suid)
        return cls.from_dict(d)

    @classmethod
    def from_DirectedGraph(cls, G, weight_attr=None, undirected=False,
                           interaction='pp'):
        """
        Create an instance from DirectedGraph instance. If weight_attr is a
        string, add the adjacency weight values as an edge attribute weight_attr.
        """

        self = object.__new__(cls)
        name_map = {}
        for i, node in enumerate(G.nodes):
            obj = self.create_node(node_id=i + 1, node_name=node)
            name_map[node] = obj[DATA][ID]

        if undirected:
            edge_func = G.outgoing_undirected_edges
        else:
            edge_func = G.outgoing_edges

        for src in G.nodes:
            src_id = name_map[src]
            for tgt, x in edge_func(src):
                tgt_id = name_map[tgt]
                self._create_edge_fast(src_id, tgt_id, interaction)
        return self

    @classmethod
    def from_edge_list_table(cls, rows, name='Edgelist Network',
                             src_col=0, tgt_col=1):

        """
        Insert node attributes from a table.

        rows is a list of rows as returned from load_edge_table
        """
        self = cls(name)
        counter = 1

        # Insert all nodes
        for edge, _ in rows:
            src_id, tgt_id = edge
            if not self.has_node(src_id) and src_id is not None:
                self.create_node(node_id=counter, node_name=src_id)
                counter += 1
            if not self.has_node(tgt_id) and tgt_id is not None:
                self.create_node(node_id=counter, node_name=tgt_id)
                counter += 1

        # Insert edges and their attributes
        for edge, data in rows:
            src_id, tgt_id = edge
            if src_id is None or tgt_id is None:
                continue
            _src_id = self._get_main_node_id(src_id)
            _tgt_id = self._get_main_node_id(tgt_id)
            obj = self.create_edge(_src_id, _tgt_id)
            for attr, item in zip(data._fields, data):
                obj[DATA][attr] = item
        return self

    def subnetwork(self, subnet_nodes, name='New Subnetwork'):
        """
        Extract a subnetwork induced by subset_nodes.

        Does not copy network attributes, but fully copies node and edge ones.
        """

        subnet_ids = set(self._get_main_node_id(node) for node in subnet_nodes)
        PG = self.__class__(name=name)
        for i, node0 in enumerate(self.elements['nodes']):
            if node0[DATA][ID] not in subnet_ids:
                continue
            node1 = PG.create_node(node_id=i + 1, node_name=node0[DATA][NAME])
            node1.update((k, v) for k, v in node0.items() if k != DATA)
            node1[DATA].update((k, v) for k, v in node0[DATA].items()
                               if k != ID)

        for edge0 in self.elements['edges']:
            src0 = edge0[DATA]['source']
            tgt0 = edge0[DATA]['target']
            if src0 in subnet_ids and tgt0 in subnet_ids:
                src1 = self.get_node_attr(src0, NAME)
                tgt1 = self.get_node_attr(tgt0, NAME)
                _src = PG._get_main_node_id(src1)
                _tgt = PG._get_main_node_id(tgt1)
                edge1 = PG.create_edge(_src, _tgt)
                edge1.update((k, v) for k, v in edge0.items() if k != DATA)
                edge1[DATA].update((k, v) for k, v in node0[DATA].items()
                                   if k not in ('source', 'target'))
        return PG

    def post_to_cytoscape(self, collection=None, host=DEFAULT_HOST,
                          port=DEFAULT_PORT, fix_col_types=False):
        """
        Post the network to Cytoscape and return network suid
        """
        client = CytoscapeClient(host, port)
        return client.save_network(self, collection, fix_col_types)

    def insert_column_types(self):

        self.columnTypes = {"node": [], "edge": [], "network": []}

        # Node attributes
        _type_map = defaultdict(set)
        for obj in self.elements['nodes']:
            for attr in obj[DATA]:
                if attr in (ID, NAME, 'SUID'):
                    continue
                val = obj[DATA][attr]
                if val is None:
                    continue
                java_type = map_property_value_type(val)[1]
                _type_map[attr].add(java_type)

        for attr, java_types in _type_map.items():
            assert len(java_types) == 1
            tobj = {'columnName': attr, 'type': java_types.pop()}
            self.columnTypes['node'].append(tobj)

        # Edge attributes
        _type_map = defaultdict(set)
        for obj in self.elements['edges']:
            for attr in obj[DATA]:
                if attr in ('source', 'target', 'SUID'):
                    continue
                val = obj[DATA][attr]
                if val is None:
                    continue
                java_type = map_property_value_type(val)[1]
                _type_map[attr].add(java_type)

        for attr, java_types in _type_map.items():
            assert len(java_types) == 1
            tobj = {'columnName': attr, 'type': java_types.pop()}
            self.columnTypes['edge'].append(tobj)

        # Network attributes
        _type_map = defaultdict(set)
        for attr in self.data:
            if attr in ('SUID', 'selected'):
                continue
            java_type = map_property_value_type(self.data[attr])[1]
            _type_map[attr].add(java_type)

        for attr, java_types in _type_map.items():
            assert len(java_types) == 1
            tobj = {'columnName': attr, 'type': java_types.pop()}
            self.columnTypes['network'].append(tobj)

    def insert_node_attributes_from_table(self, rows):
        """
        Insert node attributes from a table.

        rows is a list of rows such as returned from load_node_table
        """

        # Here we avoid the issues of proper mappings of identifiers
        for node_id, data in rows:
            if self.has_node(node_id):
                for attr, val in zip(data._fields, data):
                    self.set_node_attr(node_id, attr, val)

    def insert_node_attributes_from_dict(self, node_attrs):
        """
        Insert node attributes from a mapping node |-> dict(attr,val)
        """
        for node_id in node_attrs:
            if self.has_node(node_id):
                for attr, val in node_attrs[node_id].items():
                    self.set_node_attr(node_id, attr, val)

    def get_net_attr(self, attr):
        """
        Get network attribute
        """
        return self.data.get(attr, None)

    def set_net_attr(self, attr, val):
        """
        Set network attribute
        """
        self.data[attr] = val

    def get_name(self):
        """
        Get network name
        """
        return self.get_net_attr(NAME)

    def create_node(self, node_id, node_name=None):
        """
        Create a new node.

        We enforce that node ids have to be integers.
        """
        _node_suid = int(node_id)
        _node_id = str(node_id)
        _name = _node_id if node_name is None else node_name
        assert not self.has_node(_node_suid)
        obj = {DATA: {ID: _node_id,
                      NAME: _name},
               }
        self._nodeid2props[_node_suid] = obj
        self._nodeid2props[_node_id] = obj
        self._nodeid2props[_name] = obj
        self.elements['nodes'].append(obj)
        return obj

    def get_node_dict(self, node_id):
        """
        Get full node object (dictionary).
        """
        return self._nodeid2props.get(node_id, {})

    def get_node_attr(self, node_id, attr):
        """
        Get node attribute
        """
        node_dict = self.get_node_dict(node_id)
        if attr in node_dict:
            return node_dict[attr]
        elif DATA in node_dict:
            return node_dict[DATA].get(attr, None)
        else:
            return None

    def _get_main_node_id(self, node_id):
        return self.get_node_attr(node_id, ID)

    def set_node_attr(self, node_id, attr, val):
        """
        Set node attribute
        """
        node_dict = self.get_node_dict(node_id)
        if attr in self.node_view_attrs:
            node_dict[attr] = val
        else:
            node_dict[DATA][attr] = val

    def has_node(self, node_id):
        """
        Check if a node with node_id exists
        """
        return node_id in self._nodeid2props

    def _create_edge_fast(self, src_id, tgt_id, interaction):
        # Assumes src_id and tgt_id are canonnical ids
        obj = {DATA: {'source': src_id,
                      'target': tgt_id,
                      'interaction': interaction,
                      }
               }
        self.elements['edges'].append(obj)
        return obj

    def create_edge(self, src_id, tgt_id, interaction='reg'):
        """
        Create a new edge.

        Does not check if edge already exists
        """
        _src_id = self._get_main_node_id(src_id)
        _tgt_id = self._get_main_node_id(tgt_id)
        return self._create_edge_fast(_src_id, _tgt_id, interaction)

    def get_edge_dicts(self, src_id, tgt_id):
        """
        Get all (directed) edges between src_id and tgt_id
        """
        _src_id = self._get_main_node_id(src_id)
        _tgt_id = self._get_main_node_id(tgt_id)
        edges = []
        for obj in self.elements['edges']:
            if obj[DATA]['source'] == _src_id and \
               obj[DATA]['target'] == _tgt_id:
                edges.append(obj)
        return edges


class CytoscapeClient:
    """
    Cytoscape REST client
    """

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT):

        self.host = host
        self.port = port
        self.base = host + str(port) + '/v1'
        self.s = requests.session()

    def put(self, route, put_json):
        url = self.base + route
        response = self.s.put(url, data=put_json, headers=HEADERS)
        response.raise_for_status()
        retval = None
        try:
            retval = response.json()
        except ValueError as e:
            pass
        return retval

    def post(self, route, post_json, params=None):
        url = self.base + route
        response = self.s.post(url, data=post_json, headers=HEADERS,
                               params=params)
        response.raise_for_status()
        retval = None
        try:
            retval = response.json()
        except ValueError as e:
            pass
        return retval

    def delete(self, route):
        url = self.base + route
        response = self.s.delete(url)
        response.raise_for_status()
        retval = None
        try:
            retval = response.json()
        except ValueError as e:
            pass
        return retval

    def get(self, route, get_params=None):
        url = self.base + route
        response = self.s.get(url, params=get_params)
        response.raise_for_status()
        return response.json()

    # Style methods

    def get_all_styles(self):
        return self.get('/apply/styles')

    def apply_style(self, style_name, network_id):
        route = '/apply/styles/%s/%d' % (style_name, network_id)
        return self.get(route)

    def get_all_style_names(self):
        return self.get('/styles')

    def save_new_style(self, obj):
        res = self.post('/styles', json.dumps(obj))
        return res['title']

    def delete_all_styles(self):
        return self.delete('/styles')

    def get_num_styles(self):
        return self.get('/styles/count')

    def get_style(self, style_name):
        route = '/styles/%s' % style_name
        return self.get(route)

    def delete_style(self, style_name):
        route = '/styles/%s' % style_name
        return self.delete(route)

    # Session methods

    def save_session(self, path):
        return self.get('/session', get_params={'file': path})

    def clear_session(self, clear_styles=False):
        res = self.delete('/session')
        if clear_styles:
            self.delete_all_styles()
        return res

    def load_session(self, path):
        return self.post('/session', params={'file': path})

    def get_session_name(self):
        res = self.get('/session/name')
        return res['name']

    # Layout methods

    def get_layout_names(self):
        return self.get('/apply/layouts')

    def apply_layout(self, layout_name, network_id):
        route = '/apply/layouts/%s/%d' % (layout_name, network_id)
        return self.get(route)

    # Network methods

    def get_all_networks(self, column=None, query=None):
        params = {}
        if column is not None:
            params['column'] = column
        if query is not None:
            params['query'] = query
        return self.get('/networks.json', params)

    def get_network_suids(self, column=None, query=None):
        params = {}
        if column is not None:
            params['column'] = column
        if query is not None:
            params['query'] = query
        return self.get('/networks', params)

    def delete_all_networks(self):
        return self.delete('/networks')

    def save_network(self, PG, collection=None, fix_col_types=True):
        if fix_col_types:
            PG.insert_column_types()
        if collection is None:
            collection = PG.get_name()
        params = {'collection': collection,
                  'format': 'json'}
        res = self.post('/networks', json.dumps(PG.to_dict()), params=params)
        new_suid = res['networkSUID']
        return new_suid

    def get_network(self, network_id):
        route = '/networks/%d' % network_id
        return self.get(route)

    def delete_network(self, network_id):
        route = '/networks/%d' % network_id
        return self.delete(route)

    # Table methods

    def get_all_tables(self):
        network_id = 0
        route = '/networks/%d/tables' % network_id
        return self.get(route)

    def get_table(self, network_id, table_type):
        assert table_type in ('defaultnode', 'defaultedge', 'defaultnetwork')
        route = '/networks/%d/tables/%s' % (network_id, table_type)
        return self.get(route)

    def update_table(self, network_id, table_type, tbl):
        assert table_type in ('defaultnode', 'defaultedge', 'defaultnetwork')
        route = '/networks/%d/tables/%s' % (network_id, table_type)
        return self.put(route, json.dumps(tbl))

    def get_all_columns(self, network_id, table_type):
        assert table_type in ('defaultnode', 'defaultedge', 'defaultnetwork')
        route = '/networks/%d/tables/%s/columns' % (network_id, table_type)
        return self.get(route)

    def update_column_name(self, network_id, table_type, obj):
        assert table_type in ('defaultnode', 'defaultedge', 'defaultnetwork')
        route = '/networks/%d/tables/%s/columns' % (network_id, table_type)
        return self.put(route, json.dumps(obj))

    def create_new_column(self, network_id, table_type, obj):
        assert table_type in ('defaultnode', 'defaultedge', 'defaultnetwork')
        route = '/networks/%d/tables/%s/columns' % (network_id, table_type)
        return self.post(route, json.dumps(obj))

    def get_all_column_values(self, network_id, table_type, column_name):
        assert table_type in ('defaultnode', 'defaultedge', 'defaultnetwork')
        route = '/networks/%d/tables/%s/columns/%s' % \
                (network_id, table_type, column_name)
        return self.get(route)

    def update_column_values(self, network_id, table_type, column_name, obj):
        assert table_type in ('defaultnode', 'defaultedge', 'defaultnetwork')
        route = '/networks/%d/tables/%s/columns/%s' % \
                (network_id, table_type, column_name)
        return self.put(route, json.dumps(obj))

    def delete_column(self, network_id, table_type, column_name):
        assert table_type in ('defaultnode', 'defaultedge', 'defaultnetwork')
        route = '/networks/%d/tables/%s/columns/%s' % \
                (network_id, table_type, column_name)
        return self.delete(route)

    # View methods

    def get_first_view(self, network_id):
        route = '/networks/%d/views/first' % network_id
        return self.get(route)

    # Utility methods

    def set_style_from_file(self, style_path, network_id):
        """
        Try to import a style from a JSON file and set it for the network.

        Convention for style file names: <style_name>.sty.json.
        If style_name is present within Cytoscape session, it is applied
        without loading the file. If the file cannot be found on its specified
        path, a file with the same name is searched in module directory.
        """

        style_dir, style_file = os.path.split(style_path)
        style_name = style_file.replace('.sty.json', '')

        # Search for style
        if style_name not in self.get_all_style_names():

            # First, try to load from the specified path.
            # Otherwise, try default styles and raise exception if not found
            if not os.path.exists(style_path):
                module_dir = os.path.dirname(os.path.abspath(__file__))
                style_path = os.path.join(module_dir, 'styles',
                                          style_name + '.sty.json')

            with open(style_path, 'r') as fp:
                style_obj = json.load(fp)
            self.save_new_style(style_obj)

        # Try to apply it but fail gracefully
        try:
            self.apply_style(style_name, network_id)
        except:
            pass


def post_style_from_file(style_json_file, host=DEFAULT_HOST,
                         port=DEFAULT_PORT):
    """
    Post style from JSON file to Cytoscape
    """
    with open(style_json_file, 'r') as fp:
        data = json.loads(fp.read())
    client = CytoscapeClient(host, port)
    return client.save_new_style(data)


def post_style_from_dict(style_obj, host=DEFAULT_HOST, port=DEFAULT_PORT):
    """
    Post style object to Cytoscape
    """
    client = CytoscapeClient(host, port)
    return client.save_new_style(style_obj)


def apply_style_to_network(network_suid, style_name,
                           host=DEFAULT_HOST, port=DEFAULT_PORT):
    """
    Apply the named visual style to the cytoscape network given by network_suid
    """
    client = CytoscapeClient(host, port)
    client.apply_style(style_name, network_suid)


def apply_layout_to_network(network_suid, layout_name,
                            host=DEFAULT_HOST, port=DEFAULT_PORT):
    """
    Apply the named layout to the cytoscape network given by network_suid
    """
    client = CytoscapeClient(host, port)
    client.apply_layout(layout_name, network_suid)


def save_cytoscape_session(path, host=DEFAULT_HOST, port=DEFAULT_PORT):
    """
    Save Cytoscape session
    """
    client = CytoscapeClient(host, port)
    client.save_session(path)


def clear_cytoscape_session(host=DEFAULT_HOST, port=DEFAULT_PORT):
    """
    Delete Cytoscape session and start a new one
    """
    client = CytoscapeClient(host, port)
    client.clear_session(clear_styles=True)
