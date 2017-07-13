import unittest

import ddt
import mock

from transferlearning import helper


@ddt.ddt
class HelperTestCase(unittest.TestCase):

    def setUp(self):
        pass

        # mock.Mock(watson_online_store.os.environ, return_value={})
        # self.slack_client = mock.Mock()
        # self.conv_client = mock.Mock()
        # self.fake_workspace_id = 'fake workspace id'
        # self.conv_client.list_workspaces.return_value = {
            # 'workspaces': [{'workspace_id': self.fake_workspace_id,
                            # 'name': 'watson-online-store'}]}
        # self.cloudant_store = mock.Mock()
        # self.discovery_client = mock.Mock()
        # self.fake_data_source = 'IBM_STORE'
        # self.fake_environment_id = 'fake env id'
        # self.fake_collection_id = "fake collection id"
        # self.discovery_client.get_environment.return_value = {
            # 'environment_id': self.fake_environment_id}
        # self.discovery_client.get_environments.return_value = {
            # 'environments': [{'environment_id': self.fake_environment_id,
                             # 'name': 'ibm-logo-store'}]}
        # self.discovery_client.get_collection.return_value = {
            # 'collection_id': self.fake_collection_id}
        # self.discovery_client.list_collections.return_value = {
            # 'collections': [{'collection_id': self.fake_collection_id,
                             # 'name': 'ibm-logo-store'}]}

        # self.wos = watson_online_store.WatsonOnlineStore(
            # 'UBOTID',
            # self.slack_client,
            # self.conv_client,
            # self.discovery_client,
            # self.cloudant_store)

    def test_create_image_lists_bogus_dir(self):

        ret = helper.create_image_lists('this is not a real dir', 'ignored', 'ignored')

        self.assertIsNone(ret)
